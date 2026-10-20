"""V4: propulsion-constrained mission smoothing. Run python navigation_v4.py."""
from pathlib import Path
import argparse, hashlib, json, time
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parent
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d
import navigation_v3 as v

def smooth_motion(rr,prior,prop,motion=.03,lateral=.02,dyn=.025,timing=.5,cur=.004,model=.025,obs=.02,curacc=None,rho=.95):
 offsets=np.cumsum([0]+[4*len(r['t'])+2 for r in rr]);glob=offsets[-1]
 row=[];col=[];data=[];rhs=[];sig=[];dvlrows=[]
 def eq(ent,val,sd):
  k=len(rhs)
  for c,z in ent:row.append(k);col.append(c);data.append(z)
  rhs.append(val);sig.append(sd);return k
 for o,r in zip(offsets,rr):
  n=len(r['t']);dt=.5;acc=r['accel'];ang=r['angle'];rot=v.rotation_matrix(ang[:,0],ang[:,1],ang[:,2]);yaw=ang[:,2]
  for i in r['idx']:
   q=np.clip(r['pt'][i]/dt,0,n-1);j=min(int(q),n-2);u=q-j
   for axis in range(2):
    va=o+axis*2*n;ca=va+n;ent=[(va+j,1-u),(va+j+1,u)]
    if r['wt'][i]:ent.extend([(ca+j,-(1-u)),(ca+j+1,-u)])
    noise=obs*(1.4-.6*float(r['g'].dvl_quality.iloc[i]))
    dvlrows.append(eq(ent,r['world'][i,axis],noise))
  speed=r['g'].desired_speed_mps.to_numpy();rate=gaussian_filter1d(r['rate'],2)
  previous=None
  for j in range(n):
   co,si=np.cos(yaw[j]),np.sin(yaw[j])
   ents=[[(o+j,co),(o+n+j,-co),(o+2*n+j,si),(o+3*n+j,-si)],[(o+j,-si),(o+n+j,si),(o+2*n+j,co),(o+3*n+j,-co)]]
   targets=[prop[0]+prop[1]*speed[j],prop[2]+prop[3]*speed[j]+prop[4]*speed[j]*rate[j]]
   for axis,sd in enumerate([motion,lateral]):
    if previous is None:eq(ents[axis],targets[axis],sd)
    else:eq(ents[axis]+[(col,-rho*value) for col,value in previous[0][axis]],targets[axis]-rho*previous[1][axis],sd*np.sqrt(1-rho*rho))
   previous=(ents,targets)
  for axis in range(2):
   va=o+axis*2*n;ca=va+n
   for j in range(n-1):
    midrot=(1-timing)*rot[j,axis]+timing*rot[j+1,axis]
    ent=[(va+j,-1),(va+j+1,1),(o+4*n+axis,-dt)]
    ent.extend([(glob+2+k,-dt*midrot[k]) for k in range(3)])
    eq(ent,dt*((1-timing)*acc[j,axis]+timing*acc[j+1,axis]),dyn)
    eq([(ca+j,-1),(ca+j+1,1)],0.,cur)
    if curacc is not None and j<n-2:eq([(ca+j,1),(ca+j+1,-2),(ca+j+2,1)],0.,curacc)
   for j in r['updates']:eq([(ca+j,1),(glob+axis,-1)],r['current'][j,axis],model)
   eq([(o+4*n+axis,1)],0.,.003)
 for axis in range(2):eq([(glob+axis,1)],prior[axis],.05)
 for axis in range(3):eq([(glob+2+axis,1)],0.,.05)
 matrix=sparse.csr_matrix((data,(row,col)),shape=(len(rhs),glob+5));rhs=np.asarray(rhs);w0=1/np.square(sig);robust=np.ones(len(rhs));dvlrows=np.array(dvlrows)
 for _ in range(3):
  w=w0*robust;h=matrix.T@sparse.diags(w)@matrix;sol=spsolve(h.tocsc(),matrix.T@(w*rhs))
  res=(matrix[dvlrows]@sol-rhs[dvlrows]).reshape(-1,2)
  robust[dvlrows]=np.repeat(np.minimum(1,.045/np.maximum(np.linalg.norm(res,axis=1),1e-9)),2)
 out=[]
 for o,r in zip(offsets,rr):
  n=len(r['t']);theta=r['m'].track_heading_rad;cross=np.array([-np.sin(theta),np.cos(theta)])
  fix=np.array([r['m'].fix_north_m,r['m'].fix_east_m])@cross
  vel=np.column_stack([sol[o:o+n],sol[o+2*n:o+3*n]]);current=np.column_stack([sol[o+n:o+2*n],sol[o+3*n:o+4*n]])
  out.append({'base':fix+.5*np.sum(vel@cross),'endpoint':(vel[-1]-vel[0])@cross,'last':vel[-1]@cross,'current_delta':.5*np.sum((current-r['current'])@cross)})
 return out

def extract_motion(meta,raw,reference,prop,compass):
 configs={"r90":dict(motion=.03,lateral=.03,rho=.9)}
 rec,_=v.records(meta,raw);rec={i:v.prepare(r,compass) for i,r in rec.items()};out={};reference=reference.set_axis(meta.Id);prop=prop.set_axis(meta.Id)
 for count,(_,group) in enumerate(meta.groupby('mission_id')):
  rr=[rec[i] for i in group.Id];p=reference.loc[group.Id.iloc[0]];pr=prop.loc[group.Id.iloc[0],[f'prop_p{j}' for j in range(7)]].to_numpy()
  prior=pr[5:7]
  for k,config in configs.items():
   for i,feat in zip(group.Id,smooth_motion(rr,prior,pr,**config)):
    out.setdefault(i,{}).update({k+'_'+name:value for name,value in feat.items()})
  if (count+1)%50==0:print('Motion missions',count+1,flush=True)
 return pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True)
