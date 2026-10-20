"""Mission population-prior calibration with sensor posterior covariances."""
import numpy as np
from scipy.optimize import least_squares
import navigation_v3 as v
COLS=['bias_basis','yaw_basis','scale_basis','current_along_basis','window_current_correction','window_differential_correction']

def arrays(meta,a,ref,prefix='hier',endpoint=True,scale=None):
    n=sum(column.startswith(prefix+'_global_mean') for column in a.columns)
    sd=np.array([.008,.07,.010,.010,.06,.008,.006,.008,.006,.004]+([.008] if n>=11 else [])+([.01,.01] if n==13 else []))
    idx=np.stack(list(meta.groupby('mission_id',sort=False).indices.values()))
    m=a[[f'{prefix}_global_mean{k}' for k in range(n)]].to_numpy()[idx[:,0]]
    center=np.zeros(n);center[1]=1
    m=(m-center)/sd
    q=a[[f'{prefix}_global_cov{k}_{j}' for k in range(n) for j in range(n)]].to_numpy()[idx[:,0]].reshape(-1,n,n)/sd[None,:,None]/sd[None,None,:]
    c=a[[f'{prefix}_prior_basis{k}' for k in range(n)]].to_numpy()[idx]*sd
    z=ref[COLS].to_numpy()
    if endpoint:z=np.column_stack([z,a[prefix+'_endpoint'].to_numpy()])
    if scale is None:scale=np.maximum(np.std(z,axis=0),.001)
    return (idx,m,q,c,a[prefix+'_base'].to_numpy()[idx],z[idx]/scale),scale

def subset(arr,mask):
    idx,m,q,c,base,z=arr;flat=idx[mask];order=np.argsort(flat.ravel());newidx=np.argsort(order).reshape(flat.shape)
    return (newidx,m[mask],q[mask],c[mask],base[mask],z[mask]),flat.ravel()[order]

def predict(par,arr):
    idx,m,q,c,base,z=arr;n=m.shape[1];groups=np.array([0,1,2,3,4,5,5,5,6,7]+list(range(8,n-2)));nv=groups.max()+1
    precision=np.exp(-2*par[n:n+nv][groups]);d=precision-1
    solution=np.linalg.solve(np.eye(n)[None,:,:]+d[None,:,None]*q,(precision*par[:n]-d*m)[:,:,None])[:,:,0]
    out=np.empty(idx.size);out[idx]=base+np.einsum('gij,gj->gi',c,solution)+np.einsum('gik,k->gi',z,par[n+nv:])
    return out

def fit(arr,y,mean_reg=.5,var_reg=.4):
    n=arr[1].shape[1];nv=n-2;nz=arr[-1].shape[-1]
    penalty=np.r_[np.full(n,mean_reg),np.full(nv,var_reg),np.full(nz,.5)]
    objective=lambda par:np.r_[v.distance(predict(par,arr))-y,penalty*par]
    return least_squares(objective,np.zeros(len(penalty)),bounds=(np.r_[np.full(n,-12.),np.full(nv,-2.),np.full(nz,-10.)],np.r_[np.full(n,12.),np.full(nv,2.),np.full(nz,10.)]),max_nfev=80,diff_step=1e-4).x
