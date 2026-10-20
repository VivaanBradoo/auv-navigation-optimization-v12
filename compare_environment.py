"""Compare current priors and posterior uncertainty on identical mission folds."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import model_experiment as h,population_model as pop,navigation_v3 as v

def posterior_variance(par,arr,raw,fix):
 idx,m,q,c,base,z=arr
 groups=np.array([0,1,2,3,4,5,5,5,6,7,8]);d=np.exp(-2*par[11:20][groups])-1
 correction=np.linalg.solve(np.eye(11)[None,:,:]+d[None,:,None]*q,d[None,:,None]*c.transpose(0,2,1))
 variance=raw[idx]-np.einsum('gij,gji->gi',c,correction)+fix[idx]**2
 out=np.empty(idx.size);out[idx]=np.maximum(variance,1e-8)
 return out

def main():
 threadpool_limits(1,user_api='blas')
 tr,te,(x,f,ef),_=h.load_tables();r,*_=pd.read_pickle('v3_sensor_features.pkl');a=pd.read_pickle('environment_features.pkl');y=tr[v.TARGET].to_numpy()
 names=['initial','stationary','physical','gentle'];arrays={name:pop.arrays(tr,a,r,name,True)[0] for name in names};records={};parameters=[];ref=pd.read_pickle('fivefold_reference_predictions.pkl')['v7']
 for seed in [2026,731,1907]:
  out={name+'_'+kind:np.zeros(len(y)) for name in names for kind in ['fixed','physical']}
  for fold,(ii,jj) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,y,tr.mission_id)):
   record={'seed':seed,'fold':fold}
   for name,arr in arrays.items():
    ta,order=pop.subset(arr,np.isin(arr[0][:,0],ii));par=pop.fit(ta,y[order],.15,.4);record[name]=par.tolist();mu=pop.predict(par,arr);ff=f.copy();ff['signed']=mu
    model=h.lgb.LGBMRegressor(n_estimators=150,num_leaves=5,learning_rate=.015,min_child_samples=70,reg_lambda=30,colsample_bytree=.85,random_state=42,verbosity=-1,n_jobs=4)
    model.fit(ff.iloc[ii],np.sign(mu[ii])*y[ii]-mu[ii]);signed=mu[jj]+model.predict(ff.iloc[jj]);variance=posterior_variance(par,arr,a[name+'_displacement_variance'].to_numpy(),tr.fix_sigma_m.to_numpy())
    out[name+'_fixed'][jj]=v.distance(signed);out[name+'_physical'][jj]=v.distance(signed,np.clip(np.sqrt(variance[jj]),.12,1.2))
   parameters.append(record);print(seed,'fold',fold,flush=True)
  records[seed]=out;print(seed,{name:v.metrics(y,.375*ref[seed]+.625*pred) for name,pred in out.items()},flush=True);pd.to_pickle(records,'environment_predictions.pkl')
 results={name:v.metrics(y,np.mean([.375*ref[seed]+.625*records[seed][name] for seed in records],axis=0)) for name in out}
 print('ENSEMBLE',results,flush=True);Path('environment_results.json').write_text(json.dumps(results,indent=2));Path('environment_parameters.json').write_text(json.dumps(parameters,indent=2))
if __name__=='__main__':main()
