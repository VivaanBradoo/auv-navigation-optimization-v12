"""Check uncertainty-aware calibration on unseen complete missions."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import model_experiment as h,population_model as pop,navigation_v3 as v
from compare_environment import posterior_variance

def fit(arr,y,raw,fix,start,kind):
 penalty=np.r_[np.full(11,.15),np.full(9,.4),np.full(7,.5)];initialvar=posterior_variance(start,arr,raw,fix)
 weight=(initialvar/np.mean(initialvar))**(-.25 if kind=='weighted_half' else -.5) if kind.startswith('weighted') else np.ones(len(y));weight=np.clip(weight,.4,2.5);weight/=np.sqrt(np.mean(weight**2))
 def objective(par):
  sf=np.clip(np.sqrt(posterior_variance(par,arr,raw,fix)),.12,1.2) if kind=='physical' else .43
  error=(v.distance(pop.predict(par,arr),sf)-y)*weight
  if kind=='robust':error=np.sign(error)*np.sqrt(np.where(np.abs(error)<.4,error**2,.8*np.abs(error)-.16))
  return np.r_[error,penalty*par]
 return least_squares(objective,start,bounds=(np.r_[np.full(11,-12.),np.full(9,-2.),np.full(7,-10.)],np.r_[np.full(11,12.),np.full(9,2.),np.full(7,10.)]),max_nfev=80,diff_step=1e-4).x

def main():
 threadpool_limits(1,user_api='blas');tr,te,(x,f,ef),_=h.load_tables();r,*_=pd.read_pickle('v3_sensor_features.pkl');a=pd.read_pickle('production_features.pkl');arr,_=pop.arrays(tr,a,r,'compassyaw',True);y=tr[v.TARGET].to_numpy();pars={(p['seed'],p['fold']):np.array(p['stationary']) for p in json.load(open('environment_parameters.json'))};reference=pd.read_pickle('fivefold_reference_predictions.pkl')['v7'];records={};parameters=[];raw=a.compassyaw_displacement_variance.to_numpy();fix=tr.fix_sigma_m.to_numpy()
 for seed in [2026,731,1907]:
  out={name:np.zeros(len(y)) for name in ['physical','weighted_half','weighted','robust']}
  for fold,(ii,jj) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,y,tr.mission_id)):
   ta,order=pop.subset(arr,np.isin(arr[0][:,0],ii));record={'seed':seed,'fold':fold}
   for name in out:
    par=fit(ta,y[order],raw[order],fix[order],pars[seed,fold],name);record[name]=par.tolist();mu=pop.predict(par,arr);sf=np.clip(np.sqrt(posterior_variance(par,arr,raw,fix)),.12,1.2);ff=f.copy();ff['signed']=mu
    model=h.lgb.LGBMRegressor(n_estimators=150,num_leaves=5,learning_rate=.015,min_child_samples=70,reg_lambda=30,colsample_bytree=.85,random_state=42,verbosity=-1,n_jobs=2);model.fit(ff.iloc[ii],np.sign(mu[ii])*y[ii]-mu[ii]);out[name][jj]=v.distance(mu[jj]+model.predict(ff.iloc[jj]),sf[jj])
   parameters.append(record);print(seed,fold,flush=True)
  records[seed]=out;pd.to_pickle(records,'calibration_predictions.pkl')
 results={name:v.metrics(y,np.mean([.375*reference[s]+.625*records[s][name] for s in records],axis=0)) for name in out};print('RESULTS',results,flush=True);Path('calibration_results.json').write_text(json.dumps(results,indent=2));Path('calibration_parameters.json').write_text(json.dumps(parameters,indent=2))
if __name__=='__main__':main()
