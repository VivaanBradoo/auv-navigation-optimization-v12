"""Anisotropic quality-weighted DVL smoothing with a spike mixture likelihood."""
import pandas as pd
import navigation_v3 as v
from mixture_solver import smooth_motion

def extract_motion(meta,raw,reference,prop,compass):
    rec,_=v.records(meta,raw);rec={i:v.prepare(r,compass) for i,r in rec.items()};out={};prop=prop.set_axis(meta.Id)
    for count,(_,group) in enumerate(meta.groupby('mission_id')):
        rr=[rec[i] for i in group.Id];pr=prop.loc[group.Id.iloc[0],[f'prop_p{j}' for j in range(7)]].to_numpy()
        for i,feat in zip(group.Id,smooth_motion(rr,pr[5:7],pr,motion=.012823,lateral=.029955,rho=(.77555,.85832),bodyobs=(.018259/.98,.014055/.98),quality=True)):
            out[i]={'compassyaw_'+name:value for name,value in feat.items()}
        if (count+1)%50==0:print('Motion missions',count+1,flush=True)
    return pd.DataFrame.from_dict(out,orient='index').loc[meta.Id].reset_index(drop=True)
