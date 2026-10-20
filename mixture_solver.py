"""V4: propulsion-constrained mission smoothing. Run python navigation_v4.py."""
from pathlib import Path
import argparse, hashlib, json, time
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parent
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve,splu
from scipy.ndimage import gaussian_filter1d
import navigation_v3 as v

def smooth_motion(rr,prior,prop,motion=.03,lateral=.02,dyn=.025,timing=.5,cur=.004,model=.025,obs=.02,curacc=None,rho=(.79,.86),bodyobs=None,quality=True):
 offsets=np.cumsum([0]+[4*len(r['t'])+2 for r in rr]);glob=offsets[-1]
 row=[];col=[];data=[];rhs=[];sig=[];dvlrows=[]
 def eq(ent,val,sd):
  k=len(rhs)
  for c,z in ent:row.append(k);col.append(c);data.append(z)
  rhs.append(val);sig.append(sd);return k
 for o,r in zip(offsets,rr):
  n=len(r['t']);dt=.5;acc=r['accel'];ang=r['angle'];rot=v.rotation_matrix(ang[:,0],ang[:,1],ang[:,2]);yaw=ang[:,2]
  for i in r['idx']:
   pingents=[];pingrhs=[]
   q=np.clip(r['pt'][i]/dt,0,n-1);j=min(int(q),n-2);u=q-j
   for axis in range(2):
    va=o+axis*2*n;ca=va+n;ent=[(va+j,1-u),(va+j+1,u)]
    if r['wt'][i]:ent.extend([(ca+j,-(1-u)),(ca+j+1,-u)])
    noise=obs*(1.4-.6*float(r['g'].dvl_quality.iloc[i])) if quality else obs
    lag=np.array([np.interp(r['pt'][i],r['t'],ang[:,k]) for k in range(3)])
    rotlag=v.rotation_matrix(lag[0],lag[1],lag[2])
    for k in range(3):ent.append((glob+10+k,-rotlag[axis,k]))
    ent.append((glob+13,r['world'][i,1] if axis==0 else -r['world'][i,0]))
    ent.append((glob+14,-r['world'][i,axis]))
    ent.append((glob+15,r['world'][i,1] if axis==0 else -r['world'][i,0]))
    if bodyobs is None:dvlrows.append(eq(ent,r['world'][i,axis],noise))
    else:pingents.append(ent);pingrhs.append(r['world'][i,axis])
   if bodyobs is not None:
    co,si=np.cos(lag[2]),np.sin(lag[2]);transform=np.array([[co,si],[-si,co]])
    for axis in range(2):
     ent=[(column,transform[axis,k]*value) for k in range(2) for column,value in pingents[k]]
     noise=bodyobs[axis]*(1.4-.6*float(r['g'].dvl_quality.iloc[i])) if quality else bodyobs[axis]
     dvlrows.append(eq(ent,transform[axis]@np.array(pingrhs),noise))
  speed=r['g'].desired_speed_mps.to_numpy();rate=gaussian_filter1d(r['rate'],2)
  previous=None
  for j in range(n):
   co,si=np.cos(yaw[j]),np.sin(yaw[j])
   ents=[[(o+j,co),(o+n+j,-co),(o+2*n+j,si),(o+3*n+j,-si),(glob+5,-1),(glob+6,-speed[j])],[(o+j,-si),(o+n+j,si),(o+2*n+j,co),(o+3*n+j,-co),(glob+7,-1),(glob+8,-speed[j]),(glob+9,-speed[j]*rate[j])]]
   ents[0].append((glob+15,prop[2]+prop[3]*speed[j]+prop[4]*speed[j]*rate[j]));ents[1].append((glob+15,-(prop[0]+prop[1]*speed[j])))
   targets=[0.,0.]
   for axis,sd in enumerate([motion,lateral]):
    axisrho=rho[axis]
    if previous is None:eq(ents[axis],targets[axis],sd)
    else:eq(ents[axis]+[(col,-axisrho*value) for col,value in previous[0][axis]],targets[axis]-axisrho*previous[1][axis],sd*np.sqrt(1-axisrho*axisrho))
   previous=(ents,targets)
  for axis in range(2):
   va=o+axis*2*n;ca=va+n
   for j in range(n-1):
    midrot=(1-timing)*rot[j,axis]+timing*rot[j+1,axis]
    ent=[(va+j,-1),(va+j+1,1),(o+4*n+axis,-dt)]
    ent.extend([(glob+2+k,-dt*midrot[k]) for k in range(3)])
    midacc=(1-timing)*acc[j]+timing*acc[j+1];ent.append((glob+15,dt*midacc[1] if axis==0 else -dt*midacc[0]))
    eq(ent,dt*midacc[axis],dyn)
    eq([(ca+j,-1),(ca+j+1,1)],0.,cur)
    if curacc is not None and j<n-2:eq([(ca+j,1),(ca+j+1,-2),(ca+j+2,1)],0.,curacc)
   for j in r['updates']:eq([(ca+j,1),(glob+axis,-1)],r['current'][j,axis],model)
   eq([(o+4*n+axis,1)],0.,.003)
 for axis in range(2):eq([(glob+axis,1)],0.,.04)
 for axis in range(3):eq([(glob+2+axis,1)],0.,.05)
 for k,(center,sd) in enumerate(zip([0,1,0,0,0],[.008,.07,.010,.010,.06])):eq([(glob+5+k,1)],center,sd)
 for k,sd in enumerate([.008,.006,.008,.006,.004]):eq([(glob+10+k,1)],0.,sd)
 eq([(glob+15,1)],0.,.008)
 matrix=sparse.csr_matrix((data,(row,col)),shape=(len(rhs),glob+16));rhs=np.asarray(rhs);w0=1/np.square(sig);robust=np.ones(len(rhs));dvlrows=np.array(dvlrows)
 for _ in range(3):
  w=w0*robust;h=matrix.T@sparse.diags(w)@matrix;factor=splu(h.tocsc());sol=factor.solve(matrix.T@(w*rhs))
  res=(matrix[dvlrows]@sol-rhs[dvlrows]).reshape(-1,2)
  # Posterior probability of an ordinary ping versus broad contamination.
  # A ping affects both axes together; repeated log rows are still deduplicated.
  ordinary=np.maximum(np.mean(np.square(np.asarray(sig)[dvlrows].reshape(-1,2)),axis=1),1e-6)
  spike=.15**2;norm2=np.sum(res**2,axis=1)
  logodds=np.log(.98/.02)+np.log(spike/ordinary)-.5*norm2*(1/ordinary-1/spike)
  good=1/(1+np.exp(-np.clip(logodds,-40,40)))
  robust[dvlrows]=np.repeat(good+(1-good)*ordinary/spike,2)
 prior_sds=np.array([.008,.07,.010,.010,.06,.008,.006,.008,.006,.004,.008])
 sources=np.zeros((glob+16,11));sources[glob+5:glob+16]=np.diag(1/prior_sds**2)
 sensitivity=factor.solve(sources)
 gcov=sensitivity[glob+5:glob+16]*prior_sds[None,:]**2
 out=[]
 for o,r in zip(offsets,rr):
  n=len(r['t']);theta=r['m'].track_heading_rad;cross=np.array([-np.sin(theta),np.cos(theta)])
  fix=np.array([r['m'].fix_north_m,r['m'].fix_east_m])@cross
  vel=np.column_stack([sol[o:o+n],sol[o+2*n:o+3*n]]);current=np.column_stack([sol[o+n:o+2*n],sol[o+3*n:o+4*n]])
  out.append({**{f'global_mean{k}':sol[glob+5+k] for k in range(11)},**{f'global_cov{k}_{j}':gcov[k,j] for k in range(11) for j in range(11)},**{f'prior_basis{k}':.5*np.sum(sensitivity[o:o+n,k]*cross[0]+sensitivity[o+2*n:o+3*n,k]*cross[1]) for k in range(11)},**{f'prop{k}':sol[glob+5+k] for k in range(5)},'base':fix+.5*np.sum(vel@cross),'endpoint':(vel[-1]-vel[0])@cross,'last':vel[-1]@cross,'current_delta':.5*np.sum((current-r['current'])@cross)})
 return out

def extract_motion(meta,raw,reference,prop,compass):
 configs={'process':dict(motion=.012823,lateral=.029955,rho=(.77555,.85832)), 'noise':dict(motion=.013,lateral=.028,rho=(.79,.86),bodyobs=(.018259,.014055),quality=False), 'both':dict(motion=.012823,lateral=.029955,rho=(.77555,.85832),bodyobs=(.018259,.014055),quality=False)}
 rec,_=v.records(meta,raw);rec={i:v.prepare(r,compass) for i,r in rec.items()};out={};prop=prop.set_axis(meta.Id)
 for count,(_,group) in enumerate(meta.groupby('mission_id')):
  rr=[rec[i] for i in group.Id];pr=prop.loc[group.Id.iloc[0],[f'prop_p{j}' for j in range(7)]].to_numpy()
  for name,config in configs.items():
   for i,feat in zip(group.Id,smooth_motion(rr,pr[5:7],pr,**config)):out.setdefault(i,{}).update({name+'_'+key:value for key,value in feat.items()})
  if (count+1)%25==0:print('Motion missions',count+1,flush=True)
 return pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True)
