"""Mission-grouped comparison of calibration and residual model choices."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import navigation_v3 as v
import navigation_v4 as old
ROOT=Path(__file__).resolve().parent

def load_tables():
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    r,p,c,rt,pt,ct,_=pd.read_pickle(ROOT/'v3_sensor_features.pkl')
    m,mt=pd.read_pickle(ROOT/'v4_sensor_features.pkl')
    ar=pd.read_pickle(ROOT/'ar_motion_features.pkl');art=pd.read_pickle(ROOT/'test_ar_motion_features.pkl')
    aj=pd.read_pickle(ROOT/'ar_joint_features.pkl');ajt=pd.read_pickle(ROOT/'test_ar_joint_features.pkl')
    def build(meta,r,p,c,m,ar,aj):
        x,f=old.make_tables(meta,r,p,c,m)
        x['ar_delta']=ar.r90_base-r.base;x['joint_ar']=aj.a_base-r.base
        x['new_p2']=r.bias_basis*aj.a_prop2;x['new_p3']=r.bias_basis*meta.desired_speed_mps__mean*aj.a_prop3
        enhanced=pd.concat([f,ar.add_prefix('ar_'),aj.add_prefix('aj_'),x.drop(columns='base').add_prefix('basis_'),
                            pd.DataFrame({'best_base':aj.a_base,'reconstruction_difference':aj.a_base-ar.r90_base})],axis=1)
        return x,f,enhanced
    return train,test,build(train,r,p,c,m,ar,aj),build(test,rt,pt,ct,mt,art,ajt)

def calibrate(x,y,loss,folded=False):
    base=x.base.to_numpy();a=x.drop(columns='base').to_numpy()
    transform=(lambda mu:v.distance(mu)) if folded else np.abs
    return least_squares(lambda p:np.r_[transform(base+a@p)-y,.2*p],np.zeros(a.shape[1]),loss=loss,f_scale=.7 if loss=='soft_l1' else 1.,max_nfev=100).x

def main():
    train,test,(x,f,ef),_=load_tables();y=train[v.TARGET].to_numpy()
    choices={
      'baseline':('baseline',False,False,150,5,70,30),
      'linear':('linear',False,False,150,5,70,30),
      'soft07':('soft_l1',False,False,150,5,70,30),
      'folded_linear':('linear',True,False,150,5,70,30),
      'enhanced':('baseline',False,True,150,5,70,30),
      'enhanced_linear':('linear',False,True,150,5,70,30),
      'enhanced300':('baseline',False,True,300,7,100,50),
      'enhanced600':('baseline',False,True,600,7,100,50),
      'enhanced_shallow':('baseline',False,True,300,3,100,50),
    }
    outs={name:np.zeros(len(y)) for name in choices};signed={name:np.zeros(len(y)) for name in choices}
    for fold,(ii,jj) in enumerate(GroupKFold(5,shuffle=True,random_state=2026).split(x,y,train.mission_id)):
      calibrated={}
      for name,(loss,folded,enriched,trees,leaves,child,reg) in choices.items():
        key=(loss,folded)
        if key not in calibrated:calibrated[key]=v.fit_calibration(x.iloc[ii],y[ii]) if loss=='baseline' else calibrate(x.iloc[ii],y[ii],loss,folded)
        p=calibrated[key];mu=v.predict_signed(x,p)
        features=(ef if enriched else f).copy();features['signed']=mu
        model=lgb.LGBMRegressor(n_estimators=trees,num_leaves=leaves,learning_rate=.015,min_child_samples=child,reg_lambda=reg,colsample_bytree=.85,random_state=42,verbosity=-1,n_jobs=4)
        model.fit(features.iloc[ii],np.sign(mu[ii])*y[ii]-mu[ii])
        ss=mu[jj]+model.predict(features.iloc[jj]);signed[name][jj]=ss;outs[name][jj]=v.distance(ss)
      print('fold',fold,flush=True)
    results={name:v.metrics(y,pred) for name,pred in outs.items()}
    for name,result in sorted(results.items(),key=lambda a:a[1]['rmse_m']):print(name,result,flush=True)
    pd.to_pickle({'prediction':outs,'signed':signed},ROOT/'model_predictions.pkl')
    (ROOT/'model_results.json').write_text(json.dumps(results,indent=2))
    for name,mu in signed.items():
      print('sigma',name,{s:v.metrics(y,v.distance(mu,s))['score'] for s in [.2,.3,.4,.5,.6]},flush=True)

if __name__=='__main__':main()
