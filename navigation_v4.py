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

def smooth_motion(rr,prior,prop,motion=.03,lateral=.02,dyn=.025,timing=.5,cur=.004,model=.025,obs=.02,curacc=None):
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
  for j in range(n):
   co,si=np.cos(yaw[j]),np.sin(yaw[j])
   eq([(o+j,co),(o+n+j,-co),(o+2*n+j,si),(o+3*n+j,-si)],prop[0]+prop[1]*speed[j],motion)
   eq([(o+j,-si),(o+n+j,si),(o+2*n+j,co),(o+3*n+j,-co)],prop[2]+prop[3]*speed[j]+prop[4]*speed[j]*rate[j],lateral)
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
 configs={"weak":dict(motion=.08,lateral=.05)}
 rec,_=v.records(meta,raw);rec={i:v.prepare(r,compass) for i,r in rec.items()};out={};reference=reference.set_axis(meta.Id);prop=prop.set_axis(meta.Id)
 for count,(_,group) in enumerate(meta.groupby('mission_id')):
  rr=[rec[i] for i in group.Id];p=reference.loc[group.Id.iloc[0]];pr=prop.loc[group.Id.iloc[0],[f'prop_p{j}' for j in range(7)]].to_numpy()
  prior=pr[5:7]
  for k,config in configs.items():
   for i,feat in zip(group.Id,smooth_motion(rr,prior,pr,**config)):
    out.setdefault(i,{}).update({k+'_'+name:value for name,value in feat.items()})
  if (count+1)%50==0:print('Motion missions',count+1,flush=True)
 return pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True)


def make_tables(meta,reference,propulsion,coupled,motion):
    x,f=v.make_tables(meta,reference,propulsion,coupled)
    # Keep the old reconstruction as a calibration reference. Learn the new
    # reconstruction's weight only on the training missions of each fold.
    x['motion_delta']=motion.weak_base-reference.base
    x['endpoint']=motion.weak_endpoint
    x['prop_bias']=reference.bias_basis*propulsion.prop_p2
    x['prop_speed_bias']=reference.bias_basis*meta.desired_speed_mps__mean*propulsion.prop_p3
    x['prop_scale']=reference.yaw_basis*(propulsion.prop_p1-1)
    f=pd.concat([f,motion[['weak_base','weak_endpoint','weak_last','weak_current_delta']].add_prefix('motion_')],axis=1)
    return x,f


def original_features(train,test,use_cache):
    fingerprint=v.data_fingerprint()
    cache=ROOT/'v3_sensor_features.pkl';manifest=ROOT/'v3_feature_manifest.json'
    if use_cache and cache.exists() and manifest.exists() and json.loads(manifest.read_text())==fingerprint:
        print('Using verified v3 sensor feature cache',flush=True)
        return pd.read_pickle(cache)
    raw=pd.read_parquet(ROOT/'train_timeseries.parquet');compass=v.records(train,raw)[1]
    reference=v.extract_reference(train,raw,compass=compass)
    propulsion,_=v.extract_propulsion(train,raw,reference,compass=compass)
    coupled,_=v.extract_coupled(train,raw,reference,compass=compass)
    del raw
    raw=pd.read_parquet(ROOT/'test_timeseries.parquet')
    reference_test=v.extract_reference(test,raw,compass=compass)
    propulsion_test,_=v.extract_propulsion(test,raw,reference_test,compass=compass)
    coupled_test,_=v.extract_coupled(test,raw,reference_test,compass=compass)
    result=(reference,propulsion,coupled,reference_test,propulsion_test,coupled_test,compass)
    pd.to_pickle(result,cache);manifest.write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    return result


def fingerprint():
    names=['navigation_v4.py','navigation_v3.py','features.py','metric.py','requirements.txt',
           'train.csv','test.csv','train_timeseries.parquet','test_timeseries.parquet','sample_submission.csv']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-features',action='store_true',help='Reuse sensor features only if source/input hashes match.')
    args=parser.parse_args();start=time.perf_counter()
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');sample=pd.read_csv(ROOT/'sample_submission.csv')
    assert train.Id.is_unique and test.Id.is_unique and sample.Id.is_unique
    assert set(train.mission_id).isdisjoint(test.mission_id)
    assert set(test.Id)==set(sample.Id)
    source_hashes=fingerprint()
    reference,propulsion,coupled,reference_test,propulsion_test,coupled_test,compass=original_features(train,test,args.cached_features)
    cache=ROOT/'v4_sensor_features.pkl';manifest=ROOT/'v4_feature_manifest.json'
    if args.cached_features and cache.exists() and manifest.exists() and json.loads(manifest.read_text())==source_hashes:
        motion,motion_test=pd.read_pickle(cache)
    else:
        print('V4 training: propulsion-constrained smoothing',flush=True)
        motion=extract_motion(train,pd.read_parquet(ROOT/'train_timeseries.parquet'),reference,propulsion,compass)
        print('V4 test: same label-free reconstruction',flush=True)
        motion_test=extract_motion(test,pd.read_parquet(ROOT/'test_timeseries.parquet'),reference_test,propulsion_test,compass)
        pd.to_pickle((motion,motion_test),cache)
        manifest.write_text(json.dumps(source_hashes,indent=2),encoding='utf-8')
    x,f=make_tables(train,reference,propulsion,coupled,motion)
    z,g=make_tables(test,reference_test,propulsion_test,coupled_test,motion_test)
    assert x.columns.equals(z.columns) and f.columns.equals(g.columns)
    assert np.isfinite(x.to_numpy()).all() and np.isfinite(z.to_numpy()).all()
    y=train[v.TARGET].to_numpy();oof_by_seed={};test_predictions=[];fold_results=[]
    for seed in [2026,731,1907]:
        oof=np.zeros(len(y))
        for fold,(ii,jj) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,y,train.mission_id)):
            assert set(train.mission_id.iloc[ii]).isdisjoint(train.mission_id.iloc[jj])
            model=v.fit_model(x.iloc[ii],f.iloc[ii],y[ii])
            mu,delta=v.components(x.iloc[jj],f.iloc[jj],model);oof[jj]=v.distance(mu+delta)
            mt,dt=v.components(z,g,model);test_predictions.append(v.distance(mt+dt))
            fold_results.append({'seed':seed,'fold':fold,**v.metrics(y[jj],oof[jj])})
        oof_by_seed[seed]=oof;print('Validation',seed,v.metrics(y,oof),flush=True)
    ensemble=np.mean(list(oof_by_seed.values()),axis=0);pred=np.mean(test_predictions,axis=0)
    submission=sample[['Id']].merge(pd.DataFrame({'Id':test.Id,v.TARGET:pred}),on='Id',validate='one_to_one',sort=False)
    assert list(submission.columns)==['Id',v.TARGET] and submission.Id.equals(sample.Id)
    assert len(submission)==2000 and np.isfinite(submission[v.TARGET]).all()
    assert submission[v.TARGET].between(0,1_000_000).all()
    for name in ['submission_v4.csv','submission.csv']:
        submission.to_csv(ROOT/name,index=False,encoding='utf-8',float_format='%.10g')
    saved=pd.read_csv(ROOT/'submission_v4.csv')
    assert saved.Id.equals(sample.Id) and len(saved)==2000
    # Self-score checks the official CSV format, never hidden-test accuracy.
    assert v.score(saved.copy(),saved.copy(),'Id')==100.
    out=pd.DataFrame({'Id':train.Id,'mission_id':train.mission_id,'actual':y,'oof_prediction':ensemble})
    for seed,values in oof_by_seed.items():out[f'seed_{seed}']=values
    out.to_csv(ROOT/'validation_v4_predictions.csv',index=False)
    report={'method':'V3 sensor calibration plus weak forward/lateral propulsion constraints in the joint mission smoother, endpoint correction, and speed-dependent propulsion calibration; fifteen-model ensemble.',
            'validation':v.metrics(y,ensemble),'individual_splits':{str(s):v.metrics(y,p) for s,p in oof_by_seed.items()},'folds':fold_results,
            'previous_validation_rmse_m':.444886002059,'previous_validation_score':69.2096123,
            'previous_public_score_reported_by_user':68.640,'new_public_score':None,
            'submission_rows':len(saved),'runtime_seconds':time.perf_counter()-start,'source_hashes':source_hashes,
            'limitations':'Development estimates on the supplied training data, which informed model selection. Other splits and stress holdouts are stability checks, not an untouched evaluation. Hidden public/private scores are unknown; 80 is not demonstrated.'}
    (ROOT/'validation_v4_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINISHED',json.dumps({k:report[k] for k in ['validation','individual_splits','submission_rows','runtime_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
