"""V12: uncertainty-weighted calibration of environmental current reconstruction."""
import argparse,hashlib,importlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
import model_experiment as helper
import population_model as pop
import navigation_v3 as v
import baseline_features
from compare_environment import posterior_variance
import calibration_search
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent

def hashes(files):return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}

def features(name,train,test,r,p,rt,pt,compass,rebuild=False):
    files=[name+'_sensor.py','navigation_v3.py','features.py','train.csv','test.csv','train_timeseries.parquet','test_timeseries.parquet','v3_sensor_features.pkl']
    if name=='mixture':files+=['mixture_solver.py']
    if name=='current':files+=['mixture_solver.py']
    if name=='phase':files+=['phase_solver.py']
    if name=='frame':files+=['mixture_solver.py']
    if name=='likelihood':files+=['likelihood_solver.py']
    if name=='selected':files+=importlib.import_module('selected_sensor').CACHE_DEPENDENCIES
    if name=='production':files+=importlib.import_module('production_sensor').CACHE_DEPENDENCIES
    fingerprint=hashes(files);manifest=ROOT/(name+'_v11_manifest.json')
    valid=manifest.exists() and json.loads(manifest.read_text())==fingerprint and not rebuild
    result=[]
    for meta,ref,prop,suffix,raw in [(train,r,p,'','train_timeseries.parquet'),(test,rt,pt,'_test','test_timeseries.parquet')]:
        path=ROOT/(name+suffix+'_features.pkl')
        if valid and path.exists():a=pd.read_pickle(path)
        else:
            a=importlib.import_module(name+'_sensor').extract_motion(meta,pd.read_parquet(ROOT/raw),ref,prop,compass);a.to_pickle(path)
        assert len(a)==len(meta) and np.isfinite(a.to_numpy()).all();result.append(a)
    manifest.write_text(json.dumps(fingerprint,indent=2));return result

def tree(linear=False):
    return helper.lgb.LGBMRegressor(n_estimators=100 if linear else 150,num_leaves=5,linear_tree=linear,linear_lambda=100,learning_rate=.015,min_child_samples=100 if linear else 70,reg_lambda=30,colsample_bytree=.85,random_state=42,verbosity=-1,n_jobs=4)

def main():
    threadpool_limits(1,user_api='blas')
    parser=argparse.ArgumentParser();parser.add_argument('--folds',type=int,default=25);parser.add_argument('--new-model',default='environment_stationary',choices=['environment_stationary']);parser.add_argument('--new-weight',type=float,default=.625);parser.add_argument('--v7-weight',type=float,default=.375);parser.add_argument('--rebuild',action='store_true');parser.add_argument('--refit-core',action='store_true');args=parser.parse_args();new=args.new_model;start=time.perf_counter()
    assert 0<args.new_weight<=1 and 0<=args.v7_weight<1 and args.new_weight+args.v7_weight<=1
    baseline_features.ensure(args.rebuild)
    train,test,(x,f,ef),(xt,g,eg)=helper.load_tables();r,p,c,rt,pt,ct,compass=pd.read_pickle(ROOT/'v3_sensor_features.pkl');y=train[v.TARGET].to_numpy()
    reference={};refpath=ROOT/'v9_reference_parameters.json'
    if refpath.exists() and not args.rebuild and not args.refit_core:
        package=json.loads(refpath.read_text())
        if args.folds==package['folds'] and hashes(list(package['fingerprints']))==package['fingerprints']:
            reference={(p['seed'],p['fold']):p for p in package['parameters']}
    arrays={};test_arrays={}
    selected=None
    if (ROOT/'production_sensor.py').exists():selected=importlib.import_module('production_sensor')
    use_selected=selected is not None and selected.CANDIDATE==new
    for name in ['adaptive','mixture',new]:
        module=name.split('_')[0] if name==new else name
        prefix=name.split('_')[1] if name==new else ('hier' if name=='adaptive' else 'compassyaw');endpoint=name!='adaptive'
        if name==new and use_selected:module='production';prefix='compassyaw'
        a,at=features(module,train,test,r,p,rt,pt,compass,args.rebuild)
        arrays[name],scale=pop.arrays(train,a,r,prefix,endpoint);test_arrays[name],_=pop.arrays(test,at,rt,prefix,endpoint,scale)
        if name==new:
            new_raw=a[prefix+'_displacement_variance'].to_numpy();new_raw_test=at[prefix+'_displacement_variance'].to_numpy()
    color,colort=features('colored',train,test,r,p,rt,pt,compass,args.rebuild);cx=x.copy();cx['colored_delta']=color.rho_base-r.base;cxt=xt.copy();cxt['colored_delta']=colort.rho_base-rt.base
    oldweights={'adaptive':.5,'baseline':.25,'colored':.125,'linear':.125};weights={key:value*args.v7_weight for key,value in oldweights.items()};weights.update({'mixture':1-args.v7_weight-args.new_weight,new:args.new_weight});assert np.isclose(sum(weights.values()),1)
    weights['gaussian_new']=0.
    oofs={};previous={};previous11={};components={};test_predictions=[];parameters=[]
    for seed in [2026,731,1907]:
        out={key:np.zeros(len(y)) for key in weights}
        for fold,(ii,jj) in enumerate(GroupKFold(args.folds,shuffle=True,random_state=seed).split(x,y,train.mission_id)):
            assert set(train.mission_id.iloc[ii]).isdisjoint(train.mission_id.iloc[jj])
            mus={};mts={};record={'seed':seed,'fold':fold}
            for name in ['adaptive','mixture',new]:
                arr=arrays[name]
                if name!=new and (seed,fold) in reference:par=np.array(reference[(seed,fold)][name])
                else:ta,order=pop.subset(arr,np.isin(arr[0][:,0],ii));par=pop.fit(ta,y[order],.15,.4)
                if name==new:
                    initial=par.copy();record['initial_'+new]=initial.tolist();mus['gaussian_new']=pop.predict(initial,arr);mts['gaussian_new']=pop.predict(initial,test_arrays[name])
                    par=calibration_search.fit(ta,y[order],new_raw[order],train.fix_sigma_m.to_numpy()[order],initial,'weighted')
                mus[name]=pop.predict(par,arr);mts[name]=pop.predict(par,test_arrays[name]);record[name]=par.tolist()
            for name,table,tablet in [('baseline',x,xt),('colored',cx,cxt)]:
                par=np.array(reference[(seed,fold)][name]) if (seed,fold) in reference else helper.calibrate(table.iloc[ii],y[ii],'linear',True)
                mus[name]=v.predict_signed(table,par);mts[name]=v.predict_signed(tablet,par);record[name]=par.tolist()
            mus['linear']=mus['baseline'];mts['linear']=mts['baseline'];predt=np.zeros(len(test))
            new_sf=np.clip(np.sqrt(posterior_variance(np.array(record[new]),arrays[new],new_raw,train.fix_sigma_m.to_numpy())),.12,1.2)
            new_sf_test=np.clip(np.sqrt(posterior_variance(np.array(record[new]),test_arrays[new],new_raw_test,test.fix_sigma_m.to_numpy())),.12,1.2)
            gaussian_sf=np.clip(np.sqrt(posterior_variance(initial,arrays[new],new_raw,train.fix_sigma_m.to_numpy())),.12,1.2)
            gaussian_sf_test=np.clip(np.sqrt(posterior_variance(initial,test_arrays[new],new_raw_test,test.fix_sigma_m.to_numpy())),.12,1.2)
            for name in weights:
                linear=name=='linear';ff=(ef if linear else f).copy();ff['signed']=mus[name];gg=(eg if linear else g).copy();gg['signed']=mts[name];model=tree(linear)
                sf=new_sf[jj] if name==new else gaussian_sf[jj] if name=='gaussian_new' else .43
                sft=new_sf_test if name==new else gaussian_sf_test if name=='gaussian_new' else .43
                model.fit(ff.iloc[ii],np.sign(mus[name][ii])*y[ii]-mus[name][ii]);out[name][jj]=v.distance(mus[name][jj]+model.predict(ff.iloc[jj]),sf);predt+=weights[name]*v.distance(mts[name]+model.predict(gg),sft)
            test_predictions.append(predt);parameters.append(record)
            if (fold+1)%5==0:print(seed,'folds',fold+1,'/',args.folds,flush=True)
        components[seed]=out;oofs[seed]=sum(weights[key]*out[key] for key in weights);previous[seed]=.5*sum(oldweights[key]*out[key] for key in oldweights)+.5*out['mixture']
        previous11[seed]=.375*sum(oldweights[key]*out[key] for key in oldweights)+.625*out['gaussian_new']
        print(seed,'v12',v.metrics(y,oofs[seed]),'v11_same_splits',v.metrics(y,previous11[seed]),flush=True)
    prediction=np.mean(test_predictions,axis=0);ensemble=np.mean(list(oofs.values()),axis=0);old=np.mean(list(previous.values()),axis=0);sample=pd.read_csv(ROOT/'sample_submission.csv')
    submission=sample[['Id']].merge(pd.DataFrame({'Id':test.Id,v.TARGET:prediction}),on='Id',validate='one_to_one',sort=False)
    assert len(submission)==2000 and submission.Id.is_unique and submission.Id.equals(sample.Id) and list(submission)==['Id',v.TARGET] and np.isfinite(submission[v.TARGET]).all() and submission[v.TARGET].between(0,1e6).all()
    for filename in ['submission_v12.csv','submission.csv']:submission.to_csv(ROOT/filename,index=False,float_format='%.10g')
    old11=np.mean(list(previous11.values()),axis=0)
    pd.DataFrame({'Id':train.Id,'mission_id':train.mission_id,'actual':y,'oof_prediction':ensemble,'v9_same_splits':old,'v11_same_splits':old11,**{f'seed_{seed}':a for seed,a in oofs.items()}}).to_csv(ROOT/'validation_v12_predictions.csv',index=False);pd.to_pickle(components,ROOT/'validation_v12_components.pkl')
    sources=['navigation_v12.py','calibration_search.py','compare_environment.py','population_model.py','baseline_features.py','adaptive_sensor.py','colored_sensor.py','mixture_sensor.py','mixture_solver.py','navigation_v3.py','navigation_v4.py','model_experiment.py','motion_v5.py','joint_v5.py','features.py','metric.py','submission_v12.csv']
    if new.startswith('phase_'):sources+=['phase_solver.py']
    if new.startswith('likelihood_'):sources+=['likelihood_solver.py']
    if use_selected:sources+=['production_sensor.py','environment_solver.py']
    if reference:sources+=['v9_reference_parameters.json']
    report={'validation':v.metrics(y,ensemble),'v9_same_splits':v.metrics(y,old),'folds':args.folds,'seeds':[2026,731,1907],'individual_splits':{str(seed):v.metrics(y,pred) for seed,pred in oofs.items()},'weights':weights,'new_model':new,'reused_core_parameters':bool(reference),'private_score':65.904,'public_score':None,'previous_public_score':70.498,'rows':len(submission),'parameters':parameters,'runtime_seconds':time.perf_counter()-start,'hashes':hashes(sources),'limitations':'Private leaderboard score: 65.904. Development validation informed model selection. All comparisons use the same mission folds and training size.'}
    report['v11_same_splits']=v.metrics(y,old11)
    previous_path=ROOT/'validation_v10_predictions.csv'
    if previous_path.exists():
        previous_table=pd.read_csv(previous_path)
        assert previous_table.Id.equals(train.Id) and np.allclose(previous_table.actual,y)
        report['v10_same_splits']=v.metrics(y,previous_table.oof_prediction.to_numpy())
    previous11_path=ROOT/'validation_v11_predictions.csv'
    if previous11_path.exists():
        previous11_table=pd.read_csv(previous11_path)
        assert previous11_table.Id.equals(train.Id)
        report['v11_reproduction_max_difference']=float(np.max(np.abs(old11-previous11_table.oof_prediction.to_numpy())))
        assert report['v11_reproduction_max_difference']<1e-8
    report['previous_public_score']=70.501
    report['calibration']='Initial Gaussian fit on training missions, then fixed inverse posterior-variance weighting of training errors; residual learner unchanged.'
    report['target_71_reached']=bool(report['validation']['score']>=71)
    report['data_hashes']=hashes(['train.csv','test.csv','train_timeseries.parquet','test_timeseries.parquet','sample_submission.csv'])
    report['posterior_folding']='new component: exact conditional end-displacement variance after fold-specific prior calibration plus known fix variance; SD clipped to [0.12,1.2]'
    pd.to_pickle(test_predictions,ROOT/'validation_v12_test_predictions.pkl')
    (ROOT/'validation_v12_results.json').write_text(json.dumps(report,indent=2));print('FINISHED',report['validation'],'v11',report['v11_same_splits'],flush=True)
if __name__=='__main__':main()
