"""Selected environmental current model and exact displacement variance."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import navigation_v3 as v
from environment_solver import smooth_motion
CANDIDATE='environment_stationary'
CACHE_DEPENDENCIES=['environment_solver.py']
CURRENT_RHO=.98

def worker(job):
 threadpool_limits(1,user_api='blas');ids,rr,pr=job
 features=smooth_motion(rr,pr[5:7],pr,motion=.012823,lateral=.029955,rho=(.77555,.85832),bodyobs=(.018259/.98,.014055/.98),quality=True,cur=.008,quality_gain=1.2,initial_sd=.04,current_rho=CURRENT_RHO)
 return {ident:{'compassyaw_'+k:value for k,value in feature.items()} for ident,feature in zip(ids,features)}

def extract_motion(meta,raw,reference,prop,compass):
 rec,_=v.records(meta,raw);rec={i:v.prepare(r,compass) for i,r in rec.items()};out={};prop=prop.set_axis(meta.Id)
 co,si=np.cos(.04),np.sin(.04);rotation=np.array([[co,-si],[si,co]])
 for r in rec.values():
  r['angle'][:,2]+=.04;r['world']=r['world']@rotation.T;r['accel']=r['accel']@rotation.T;r['g']=r['g'][['dvl_quality','desired_speed_mps']].copy(deep=True)
  for key,value in list(r.items()):
   if isinstance(value,np.ndarray):r[key]=value.copy()
 del raw
 jobs=[(g.Id.tolist(),[rec[i] for i in g.Id],prop.loc[g.Id.iloc[0],[f'prop_p{k}' for k in range(7)]].to_numpy()) for _,g in meta.groupby('mission_id')]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for count,result in enumerate(pool.map(worker,jobs),1):
   out.update(result)
   if count%25==0:print('Production missions',count,flush=True)
 return pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True)
