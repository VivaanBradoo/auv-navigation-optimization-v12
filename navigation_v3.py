"""Version 3: shared-compass, propulsion-constrained inertial navigation.
All sensor feature extraction is label-free; label calibration is mission-grouped.
"""
from pathlib import Path
import argparse, json, hashlib, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import erf
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from features import rotation_matrix
from metric import navigation_score, score
ROOT=Path(__file__).resolve().parent
TARGET='cross_track_error_m'
CONFIGS={'q':dict(dyn=.025,cur=.004,model=.025,localbias=.003,quality=True)}
BASIS=['bias_basis', 'scale_basis', 'yaw_basis', 'current_n_basis', 'current_e_basis', 'yaw_sin_basis', 'yaw_cos_basis', 'bias_correction', 'rate_correction', 'current_correction', 'current_along_basis', 'window_current_correction', 'window_differential_correction', 'transition_correction', 'local_transition_correction']


def robust_linear(a,b,prior,center=None):
    center=np.zeros(a.shape[1]) if center is None else center
    penalty=np.diag(prior)
    p=center.copy()
    for _ in range(8):
        e=b-a@p
        w=np.minimum(1,.025/np.maximum(abs(e),1e-9))
        p=np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w*b)+penalty@center)
    return p

def records(meta,raw):
    lookup=meta.set_index('Id'); records={}; gyA=[];gyB=[]
    for ident,g in raw.groupby('Id',sort=False):
        g=g.sort_values('step');m=lookup.loc[ident];t=g.t_s.to_numpy()
        angle=g[['roll_rad','pitch_rad','yaw_rad']].to_numpy(copy=True)
        angle[:,2]=np.unwrap(angle[:,2]);yaw=angle[:,2]
        rate=(np.sin(angle[:,0])*g.gyro_y_radps.to_numpy()+np.cos(angle[:,0])*g.gyro_z_radps.to_numpy())/np.cos(angle[:,1])
        # Integrated gyro constrains compass distortion while demeaning removes rate bias.
        h=np.column_stack([np.sin(yaw),np.cos(yaw),np.sin(2*yaw),np.cos(2*yaw),np.sin(3*yaw),np.cos(3*yaw)])
        cum=np.r_[0,np.cumsum((rate[1:]+rate[:-1])*.25)]
        width=10
        aa=h[width:]-h[:-width]; bb=cum[width:]-cum[:-width]-(yaw[width:]-yaw[:-width])
        gyA.append(aa-aa.mean(axis=0));gyB.append(bb-bb.mean())
        records[ident]={'g':g,'m':m,'t':t,'angle':angle,'rate':rate}
    a=np.concatenate(gyA);b=np.concatenate(gyB)
    coeff=np.linalg.lstsq(a,b,rcond=None)[0]
    print('Compass correction from gyro only:',coeff,flush=True)
    return records,coeff

def extract_reference(meta,raw,compass=None):
    rec,coeff=records(meta,raw)
    if compass is not None:coeff=compass
    for ident,r in rec.items():
        g,m,t,angle=r['g'],r['m'],r['t'],r['angle']
        yaw=angle[:,2]
        angle[:,2]+=np.column_stack([np.sin(yaw),np.cos(yaw),np.sin(2*yaw),np.cos(2*yaw),np.sin(3*yaw),np.cos(3*yaw)])@coeff
        age=g.dvl_age_s.fillna(0).to_numpy();pt=t-age
        lag=np.column_stack([np.interp(pt,t,angle[:,j]) for j in range(3)])
        temp=g.water_temp_c.to_numpy()
        sound=(1449.2+4.6*temp-.055*temp**2+.00029*temp**3+.016*g.depth_m.to_numpy())/1500
        body=g[['dvl_x_mps','dvl_y_mps','dvl_z_mps']].to_numpy()*sound[:,None]
        # De-tilt into a horizontal frame whose forward direction is compass yaw.
        v=np.einsum('nij,nj->ni',rotation_matrix(lag[:,0],lag[:,1],np.zeros(len(g))),body)
        mode=g.dvl_mode.to_numpy();bt=mode==1;wt=mode==2
        cn=np.interp(pt,t,g.current_n_est_mps);ce=np.interp(pt,t,g.current_e_est_mps)
        cy,sy=np.cos(lag[:,2]),np.sin(lag[:,2])
        calong=cn*cy+ce*sy;ccross=-cn*sy+ce*cy
        vx=v[:,0]-bt*calong
        rate=np.interp(pt,t,gaussian_filter1d(r['rate'],2))
        design=np.column_stack([np.ones(len(g)),vx,bt*(-sy),bt*cy,vx*rate])
        target=v[:,1]-bt*ccross
        valid=(g.dvl_valid.to_numpy()==1)&np.isfinite(v).all(axis=1)
        idx=np.flatnonzero(valid)
        _,unique=np.unique(np.round(pt[idx],3),return_index=True);idx=idx[unique]
        err=target[idx]-median_filter(target[idx],size=7,mode='reflect')
        good=abs(err)<.15
        ground=np.column_stack([v[:,0]*cy-v[:,1]*sy+wt*cn,v[:,0]*sy+v[:,1]*cy+wt*ce])
        spatial=ground[idx]-median_filter(ground[idx],size=(7,1),mode='reflect')
        good &= np.linalg.norm(spatial,axis=1)<.3
        idx=idx[good]
        r.update(v=v,pt=pt,idx=idx,design=design,target=target,cn=cn,ce=ce,cy=cy,sy=sy,bt=bt,wt=wt,lag=lag)
        # A mode switch changes the DVL reference frame, not physical velocity.
        # Accelerometer integration accounts for actual motion between pings.
        accel=np.einsum('nij,nj->ni',rotation_matrix(angle[:,0],angle[:,1],angle[:,2]),g[['accel_x_mps2','accel_y_mps2','accel_z_mps2']].to_numpy())[:,:2]
        cumulative=np.vstack([np.zeros(2),np.cumsum((accel[1:]+accel[:-1])*.25,axis=0)])
        ia=np.column_stack([np.interp(pt[idx],t,cumulative[:,j]) for j in range(2)])
        dw=np.diff(wt[idx].astype(float));dt=np.diff(pt[idx])
        use=(dw!=0)&(dt<3.1)
        delta=-(np.diff(ground[idx],axis=0)-np.diff(ia,axis=0))[use]/dw[use,None]
        delta=delta[np.linalg.norm(delta,axis=1)<.35]
        r['transition_delta']=delta
    params={}
    for mission,group in meta.groupby('mission_id'):
        rr=[rec[i] for i in group.Id]
        a=np.concatenate([r['design'][r['idx']] for r in rr]);b=np.concatenate([r['target'][r['idx']] for r in rr])
        # Weak, physically interpretable regularization resolves near-collinearity.
        p=robust_linear(a,b,[1.,.25,.05,.05,.1])
        params[mission]=p
        deltas=np.concatenate([r['transition_delta'] for r in rr],axis=0)
        if len(deltas):
            estimate=np.median(deltas,axis=0)
            for r in rr:r['mission_transition']=estimate;r['mission_transition_count']=len(deltas)
        else:
            for r in rr:r['mission_transition']=p[2:4];r['mission_transition_count']=0
    rows=[]
    for ident,r in rec.items():
        g,m,t=r['g'],r['m'],r['t']; p=params[m.mission_id]
        idx=r['idx']; pt=r['pt'][idx]
        def integ(values):return .5*np.interp(t,pt,np.asarray(values)[idx]).sum()
        theta=m.track_heading_rad;cc,ss=np.cos(theta),np.sin(theta)
        fix=-ss*m.fix_north_m+cc*m.fix_east_m
        v=r['v'];cy,sy=r['cy'],r['sy'];wt=r['wt']
        relsin=sy*cc-cy*ss;relcos=cy*cc+sy*ss
        lateral=r['design'][:,0]*p[0]+r['design'][:,1]*p[1]
        current_cross=-(r['cn']+p[2])*ss+(r['ce']+p[3])*cc
        dvl_cross=v[:,0]*relsin+(v[:,1]-lateral)*relcos
        # Fill dropouts in through-water coordinates, preserving observed turns.
        # Interpolating world velocity across long gaps erases avoidance manoeuvres.
        corrected_n=r['cn']+p[2];corrected_e=r['ce']+p[3]
        ux=v[:,0]-r['bt']*(corrected_n*cy+corrected_e*sy)
        uy=v[:,1]-r['bt']*(-corrected_n*sy+corrected_e*cy)
        rate_ping=np.interp(r['pt'],t,gaussian_filter1d(r['rate'],2))
        yaw=r['angle'][:,2];sinrel=np.sin(yaw-theta);cosrel=np.cos(yaw-theta)
        ux_full=np.interp(t,pt,ux[idx])
        uy_full=np.interp(t,pt,(uy-p[4]*ux*rate_ping)[idx])+p[4]*ux_full*gaussian_filter1d(r['rate'],2)
        cn_full=g.current_n_est_mps.to_numpy()+p[2];ce_full=g.current_e_est_mps.to_numpy()+p[3]
        current_full=-ss*cn_full+cc*ce_full
        fullcross=ux_full*sinrel+uy_full*cosrel+current_full
        fullalong=ux_full*cosrel-uy_full*sinrel
        f={'Id':ident,'fix':fix,'base':fix+.5*fullcross.sum(),'dvl_cross':.5*(ux_full*sinrel+uy_full*cosrel).sum(),'dvl_along':.5*fullalong.sum(),'current_disp':.5*current_full.sum(),'bias_correction':integ(lateral*relcos),'rate_correction':.5*(p[4]*ux_full*gaussian_filter1d(r['rate'],2)*cosrel).sum(),'current_correction':integ(wt*(-p[2]*ss+p[3]*cc)),'duration':m.duration_s,'fix_sigma':m.fix_sigma_m,'temp':g.water_temp_c.mean(),'water_fraction':wt.mean(),'valid_fraction':len(idx)/len(g)*2}
        for j,value in enumerate(p):f[f'mission_p{j}']=value
        f['bias_basis']=.5*cosrel.sum()
        f['scale_basis']=.5*(fullcross-current_full).sum()
        f['yaw_basis']=.5*fullalong.sum()
        f['current_n_basis']=integ(wt*(-ss))
        f['current_e_basis']=integ(wt*cc)
        f['yaw_sin_basis']=.5*(fullalong*np.sin(yaw)).sum()
        f['yaw_cos_basis']=.5*(fullalong*np.cos(yaw)).sum()
        f['current_along_basis']=.5*(cn_full*cc+ce_full*ss).sum()
        residual=r['target']-r['design']@p
        ib=idx[r['bt'][idx]];iw=idx[r['wt'][idx]]
        bottom_err=np.median(residual[ib]) if len(ib) else 0.
        water_err=np.median(residual[iw]) if len(iw) else 0.
        f['window_current_correction']=bottom_err*integ(wt*relcos)*len(ib)/(len(ib)+10)
        f['window_differential_correction']=(bottom_err-water_err)*integ(wt*relcos)*len(ib)/(len(ib)+10)*len(iw)/(len(iw)+5)
        f['max_gap']=np.diff(np.r_[0,pt,m.duration_s]).max()
        trans=r['mission_transition']
        f['transition_count']=r['mission_transition_count']
        f['transition_correction']=integ(wt*(-(trans[0]-p[2])*ss+(trans[1]-p[3])*cc))
        f['transition_n']=trans[0];f['transition_e']=trans[1]
        if len(r['transition_delta']):
            local=np.median(r['transition_delta'],axis=0)
            f['local_transition_correction']=integ(wt*(-(local[0]-trans[0])*ss+(local[1]-trans[1])*cc))*len(r['transition_delta'])/(len(r['transition_delta'])+5)
        else:f['local_transition_correction']=0.
        f['fit_residual']=np.std((r['target']-r['design']@p)[idx])
        rows.append(f)
    return pd.DataFrame(rows).set_index('Id').loc[meta.Id].reset_index(drop=True)

def prepare(r,coeff):
    g,t,angle=r['g'],r['t'],r['angle'];yaw=angle[:,2].copy();n=len(t)
    angle[:,2]+=np.column_stack([np.sin(yaw),np.cos(yaw),np.sin(2*yaw),np.cos(2*yaw),np.sin(3*yaw),np.cos(3*yaw)])@coeff
    age=g.dvl_age_s.fillna(0).to_numpy();pt=t-age
    lag=np.column_stack([np.interp(pt,t,angle[:,j]) for j in range(3)])
    temp=g.water_temp_c.to_numpy();sound=(1449.2+4.6*temp-.055*temp**2+.00029*temp**3+.016*g.depth_m.to_numpy())/1500
    body=g[['dvl_x_mps','dvl_y_mps','dvl_z_mps']].to_numpy()*sound[:,None]
    world=np.einsum('nij,nj->ni',rotation_matrix(lag[:,0],lag[:,1],lag[:,2]),body)[:,:2]
    accel=np.einsum('nij,nj->ni',rotation_matrix(angle[:,0],angle[:,1],angle[:,2]),g[['accel_x_mps2','accel_y_mps2','accel_z_mps2']].to_numpy())[:,:2]
    current=g[['current_n_est_mps','current_e_est_mps']].to_numpy()
    valid=(g.dvl_valid.to_numpy()==1)&np.isfinite(world).all(axis=1)
    idx=np.flatnonzero(valid);_,unique=np.unique(np.round(pt[idx],3),return_index=True);idx=idx[unique]
    wt=g.dvl_mode.to_numpy()==2
    ground=world+wt[:,None]*current
    err=ground[idx]-median_filter(ground[idx],size=(7,1),mode='reflect')
    idx=idx[np.linalg.norm(err,axis=1)<.35]
    updates=np.r_[0,np.flatnonzero(np.diff(g.current_est_age_s.to_numpy())<0)+1]
    r.update(pt=pt,world=world,accel=accel,current=current,idx=idx,wt=wt,updates=updates)
    return r

def extract_propulsion(meta,raw,old,compass=None):
    rec,estimated=records(meta,raw);coeff=estimated if compass is None else compass
    rec={i:prepare(r,coeff) for i,r in rec.items()};out={}
    for mission,group in meta.groupby('mission_id'):
        xx=[];yy=[]
        for ident in group.Id:
            r=rec[ident];g=r['g'];idx=r['idx'];yaw=np.interp(r['pt'],r['t'],r['angle'][:,2]);cy=np.cos(yaw);sy=np.sin(yaw)
            world=r['world'];curr=r['current'];bt=(g.dvl_mode.to_numpy()==1).astype(float)
            u=g.desired_speed_mps.to_numpy();rate=np.interp(r['pt'],r['t'],gaussian_filter1d(r['rate'],2))
            vx=world[:,0]*cy+world[:,1]*sy-bt*(curr[:,0]*cy+curr[:,1]*sy)
            vy=-world[:,0]*sy+world[:,1]*cy-bt*(-curr[:,0]*sy+curr[:,1]*cy)
            ax=np.column_stack([np.ones(len(g)),u,np.zeros((len(g),3)),bt*cy,bt*sy])
            ay=np.column_stack([np.zeros((len(g),2)),np.ones(len(g)),u,u*rate,bt*(-sy),bt*cy])
            xx.extend([ax[idx],ay[idx]]);yy.extend([vx[idx],vy[idx]])
        a=np.concatenate(xx);b=np.concatenate(yy)
        p=robust_linear(a,b,[3.,.1,1.,.25,.1,.02,.02],center=np.array([0,1,0,0,0,0,0.]))
        residual=b-a@p
        for ident in group.Id:out[ident]={**{f'prop_p{j}':v for j,v in enumerate(p)},'prop_residual':np.sqrt(np.mean(np.minimum(residual**2,.1**2)))}
    f=pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True)
    f['propulsion_correction']=old.current_n_basis*(f.prop_p5-old.mission_p2)+old.current_e_basis*(f.prop_p6-old.mission_p3)
    f['propulsion_vs_transition']=old.current_n_basis*(f.prop_p5-old.transition_n)+old.current_e_basis*(f.prop_p6-old.transition_e)
    return f,coeff

def smooth_mission(rr,prior,dyn=.015,cur=.003,obs=.02,model=.025,timing=.5,localbias=.003,quality=False):
    offsets=np.cumsum([0]+[4*len(r['t'])+2 for r in rr]);glob=offsets[-1]
    # Global unknowns: current-error N/E, accelerometer body biases x/y/z.
    row=[];col=[];data=[];rhs=[];sig=[];dvlrows=[]
    def eq(ent,val,sd):
        k=len(rhs)
        for c,v in ent:row.append(k);col.append(c);data.append(v)
        rhs.append(val);sig.append(sd);return k
    for o,r in zip(offsets,rr):
        n=len(r['t']);dt=.5;acc=r['accel'];ang=r['angle']
        rot=rotation_matrix(ang[:,0],ang[:,1],ang[:,2])
        for i in r['idx']:
            q=np.clip(r['pt'][i]/dt,0,n-1);j=min(int(q),n-2);u=q-j
            for axis in range(2):
                va=o+axis*2*n;ca=va+n
                ent=[(va+j,1-u),(va+j+1,u)]
                if r['wt'][i]:ent.extend([(ca+j,-(1-u)),(ca+j+1,-u)])
                noise=obs*(1.4-.6*float(r['g'].dvl_quality.iloc[i])) if quality else obs
                dvlrows.append(eq(ent,r['world'][i,axis],noise))
        for axis in range(2):
            va=o+axis*2*n;ca=va+n
            for j in range(n-1):
                midrot=(1-timing)*rot[j,axis]+timing*rot[j+1,axis]
                ent=[(va+j,-1),(va+j+1,1),(o+4*n+axis,-dt)]
                ent.extend([(glob+2+k,-dt*midrot[k]) for k in range(3)])
                eq(ent,dt*((1-timing)*acc[j,axis]+timing*acc[j+1,axis]),dyn)
                eq([(ca+j,-1),(ca+j+1,1)],0.,cur)
            for j in r['updates']:eq([(ca+j,1),(glob+axis,-1)],r['current'][j,axis],model)
            eq([(o+4*n+axis,1)],0.,localbias)
    for axis in range(2):eq([(glob+axis,1)],prior[axis],.05)
    for axis in range(3):eq([(glob+2+axis,1)],0.,.05)
    matrix=sparse.csr_matrix((data,(row,col)),shape=(len(rhs),glob+5))
    rhs=np.asarray(rhs);w0=1/np.square(sig);robust=np.ones(len(rhs));dvlrows=np.array(dvlrows)
    for _ in range(3):
        w=w0*robust;h=matrix.T@sparse.diags(w)@matrix
        sol=spsolve(h.tocsc(),matrix.T@(w*rhs))
        res=(matrix[dvlrows]@sol-rhs[dvlrows]).reshape(-1,2)
        rrweight=np.minimum(1,.045/np.maximum(np.linalg.norm(res,axis=1),1e-9))
        robust[dvlrows]=np.repeat(rrweight,2)
    out=[]
    for o,r in zip(offsets,rr):
        n=len(r['t']);theta=r['m'].track_heading_rad;cross=np.array([-np.sin(theta),np.cos(theta)])
        fix=np.array([r['m'].fix_north_m,r['m'].fix_east_m])@cross
        v=np.column_stack([sol[o:o+n],sol[o+2*n:o+3*n]])
        c=np.column_stack([sol[o+n:o+2*n],sol[o+3*n:o+4*n]])
        out.append({'base':fix+.5*np.sum(v@cross),'mission_delta_n':sol[glob],'mission_delta_e':sol[glob+1],'accbias_x':sol[glob+2],'accbias_y':sol[glob+3],'accbias_z':sol[glob+4],'current_delta':.5*np.sum((c-r['current'])@cross)})
    return out

def extract_coupled(meta,raw,old,compass=None):
    rec,estimated=records(meta,raw);coeff=estimated if compass is None else compass
    rec={i:prepare(r,coeff) for i,r in rec.items()};out={};old=old.set_axis(meta.Id)
    for count,(mission,group) in enumerate(meta.groupby('mission_id')):
        rr=[rec[i] for i in group.Id];p=old.loc[group.Id.iloc[0]]
        prior=.4*p[['mission_p2','mission_p3']].to_numpy()+.6*p[['transition_n','transition_e']].to_numpy()
        for k,config in CONFIGS.items():
            result=smooth_mission(rr,prior,**config)
            for i,f in zip(group.Id,result):out.setdefault(i,{}).update({k+'_'+name:value for name,value in f.items()})
        if (count+1)%100==0:print('Coupled missions',count+1,'/',meta.mission_id.nunique(),flush=True)
    return pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True),coeff

def fit_calibration(x,y):
    base=x.base.to_numpy();a=x.drop(columns='base').to_numpy()
    return least_squares(lambda p:np.r_[abs(base+a@p)-y,.2*p],np.zeros(a.shape[1]),loss='soft_l1',f_scale=.35,max_nfev=100).x

def predict_signed(x,p):return x.base.to_numpy()+x.drop(columns='base').to_numpy()@p

def make_tables(meta,reference,propulsion,coupled):
    x=reference[BASIS+['base']].copy()
    x['prop']=propulsion.propulsion_correction
    x['joint_delta']=coupled.q_base-reference.base
    features=pd.concat([
        meta.drop(columns=['Id','mission_id',TARGET,'fix_north_m','fix_east_m','track_heading_rad'],errors='ignore'),
        reference.add_prefix('old_'),propulsion,
        coupled.rename(columns=lambda c:c.split('_',1)[1]).add_prefix('coupled_')],axis=1)
    cols=['accel_x_mps2__mean','accel_y_mps2__mean','gyro_x_radps__mean','gyro_y_radps__mean','gyro_z_radps__mean','dvl_quality__mean','water_temp_c__mean','motor_current_a__mean']
    agg=pd.DataFrame({c+'_mission_mean':features[c].groupby(meta.mission_id).transform('mean') for c in cols})
    return x,pd.concat([features,agg],axis=1)

def fit_model(x,f,y):
    p=fit_calibration(x,y);signed=predict_signed(x,p)
    features=f.copy();features['signed']=signed
    model=lgb.LGBMRegressor(n_estimators=150,num_leaves=5,learning_rate=.015,min_child_samples=70,reg_lambda=30,colsample_bytree=.85,random_state=42,verbosity=-1,n_jobs=4)
    model.fit(features,np.sign(signed)*y-signed)
    return p,model

def components(x,f,trained):
    p,model=trained;signed=predict_signed(x,p)
    features=f.copy();features['signed']=signed
    return signed,model.predict(features)

def distance(mu,sigma=.43):
    return abs(mu)*erf(abs(mu)/(np.sqrt(2)*sigma))+sigma*np.sqrt(2/np.pi)*np.exp(-mu**2/(2*sigma**2))

def evaluate(x,f,y,groups,seed=2026):
    physical=np.zeros(len(y));residual=physical.copy()
    for ii,jj in GroupKFold(5,shuffle=True,random_state=seed).split(x,y,groups):
        assert set(groups.iloc[ii]).isdisjoint(groups.iloc[jj])
        trained=fit_model(x.iloc[ii],f.iloc[ii],y[ii])
        physical[jj],residual[jj]=components(x.iloc[jj],f.iloc[jj],trained)
    return physical,residual

def metrics(y,pred):
    return {'rmse_m':float(np.sqrt(np.mean((y-pred)**2))),'score':navigation_score(y,pred)}

def data_fingerprint():
    names=['train.csv','test.csv','train_timeseries.parquet','test_timeseries.parquet','sample_submission.csv','features.py','navigation_v3.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-features',action='store_true',help='Reuse features only when input and source hashes match.')
    args=parser.parse_args();started=time.perf_counter()
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');sample=pd.read_csv(ROOT/'sample_submission.csv')
    assert train.Id.is_unique and test.Id.is_unique
    assert set(train.mission_id).isdisjoint(test.mission_id)
    assert set(sample.Id)==set(test.Id)
    fingerprint=data_fingerprint();cache=ROOT/'v3_sensor_features.pkl';manifest=ROOT/'v3_feature_manifest.json'
    if args.cached_features and cache.exists() and manifest.exists() and json.loads(manifest.read_text())==fingerprint:
        reference,propulsion,coupled,reference_test,propulsion_test,coupled_test,compass=pd.read_pickle(cache)
    else:
        raw=pd.read_parquet(ROOT/'train_timeseries.parquet')
        compass=records(train,raw)[1]
        print('Training: reference sensor calibration',flush=True)
        reference=extract_reference(train,raw,compass=compass)
        print('Training: propulsion constraints',flush=True)
        propulsion,_=extract_propulsion(train,raw,reference,compass=compass)
        print('Training: joint quality-weighted smoothing',flush=True)
        coupled,_=extract_coupled(train,raw,reference,compass=compass)
        del raw
        raw=pd.read_parquet(ROOT/'test_timeseries.parquet')
        print('Test: applying the same compass calibration',flush=True)
        reference_test=extract_reference(test,raw,compass=compass)
        propulsion_test,_=extract_propulsion(test,raw,reference_test,compass=compass)
        coupled_test,_=extract_coupled(test,raw,reference_test,compass=compass)
        del raw
        pd.to_pickle((reference,propulsion,coupled,reference_test,propulsion_test,coupled_test,compass),cache)
        manifest.write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    x,f=make_tables(train,reference,propulsion,coupled)
    z,g=make_tables(test,reference_test,propulsion_test,coupled_test)
    assert x.columns.equals(z.columns) and f.columns.equals(g.columns)
    assert np.isfinite(x.to_numpy()).all() and np.isfinite(z.to_numpy()).all()
    y=train[TARGET].to_numpy();oof_by_seed={};test_predictions=[];fold_results=[]
    for seed in [2026,731,1907]:
        oof=np.zeros(len(y))
        for fold,(ii,jj) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,y,train.mission_id)):
            assert set(train.mission_id.iloc[ii]).isdisjoint(train.mission_id.iloc[jj])
            trained=fit_model(x.iloc[ii],f.iloc[ii],y[ii])
            mu,residual=components(x.iloc[jj],f.iloc[jj],trained)
            oof[jj]=distance(mu+residual)
            mt,rt=components(z,g,trained);test_predictions.append(distance(mt+rt))
            fold_results.append({'seed':seed,'fold':fold,**metrics(y[jj],oof[jj])})
        oof_by_seed[seed]=oof
        print('Validation',seed,metrics(y,oof),flush=True)
    ensemble_oof=np.mean(list(oof_by_seed.values()),axis=0)
    pred=np.mean(test_predictions,axis=0)
    submission=sample[['Id']].merge(pd.DataFrame({'Id':test.Id,TARGET:pred}),on='Id',validate='one_to_one',sort=False)
    assert list(submission.columns)==['Id',TARGET]
    assert submission.Id.equals(sample.Id) and len(submission)==2000
    assert np.isfinite(submission[TARGET]).all() and submission[TARGET].between(0,1_000_000).all()
    for name in ['submission_v3.csv','submission.csv']:
        submission.to_csv(ROOT/name,index=False,encoding='utf-8',float_format='%.10g')
    saved=pd.read_csv(ROOT/'submission_v3.csv',encoding='utf-8')
    assert saved.Id.equals(sample.Id) and len(saved)==2000
    # Official metric format check only; this does not measure hidden-test accuracy.
    assert score(saved.copy(),saved.copy(),'Id')==100.
    out=pd.DataFrame({'Id':train.Id,'mission_id':train.mission_id,'actual':y,'oof_prediction':ensemble_oof})
    for seed,values in oof_by_seed.items():out[f'seed_{seed}']=values
    out.to_csv(ROOT/'validation_v3_predictions.csv',index=False)
    report={'method':'Shared compass calibration; propulsion constraints; quality-weighted joint velocity/current smoothing with shared body accelerometer bias; robust physical calibration; small residual model; fifteen-model ensemble.',
            'validation':metrics(y,ensemble_oof),'individual_splits':{str(s):metrics(y,p) for s,p in oof_by_seed.items()},'folds':fold_results,
            'previous_validation_rmse_m':.49977376409924806,'previous_validation_score':66.67672311234168,'previous_public_score_reported_by_user':63.493,'new_public_score':None,
            'submission_rows':len(saved),'compass_coefficients':compass.tolist(),'runtime_seconds':time.perf_counter()-started,
            'limitations':'All validation uses the supplied training dataset, which informed model selection. Alternate splits are stability checks, not an untouched test set. Hidden test accuracy and additional unpublished Rules-page limits are unknown.',
            'source_hashes':fingerprint}
    (ROOT/'validation_v3_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINISHED',json.dumps({k:report[k] for k in ['validation','individual_splits','submission_rows','runtime_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
