"""Audit final submission, held-out predictions and measured comparison."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
import metric
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report=json.loads((ROOT/'validation_v12_results.json').read_text());sample=pd.read_csv(ROOT/'sample_submission.csv');test=pd.read_csv(ROOT/'test.csv');train=pd.read_csv(ROOT/'train.csv');csv=pd.read_csv(ROOT/'submission_v12.csv');oof=pd.read_csv(ROOT/'validation_v12_predictions.csv')
assert csv.Id.equals(sample.Id) and csv.Id.equals(test.Id) and csv.Id.is_unique and len(csv)==2000
assert list(csv)==['Id',metric.TARGET] and np.isfinite(csv[metric.TARGET]).all() and csv[metric.TARGET].between(0,1e6).all()
assert (ROOT/'submission_v12.csv').read_bytes()==(ROOT/'submission.csv').read_bytes()
assert oof.Id.equals(train.Id) and np.allclose(oof.actual,train[metric.TARGET]) and oof.mission_id.equals(train.mission_id)
score=metric.navigation_score(oof.actual.to_numpy(),oof.oof_prediction.to_numpy());assert abs(score-report['validation']['score'])<1e-10
assert all(sha(ROOT/name)==value for name,value in report['hashes'].items())
assert all(sha(ROOT/name)==value for name,value in report['data_hashes'].items())
components=pd.read_pickle(ROOT/'validation_v12_components.pkl');combined=np.mean([sum(report['weights'][key]*pred for key,pred in record.items()) for record in components.values()],axis=0)
assert np.max(np.abs(combined-oof.oof_prediction.to_numpy()))<1e-12
test_predictions=np.mean(pd.read_pickle(ROOT/'validation_v12_test_predictions.pkl'),axis=0);assert np.allclose(test_predictions,csv[metric.TARGET],rtol=1e-9,atol=1e-9)
if (ROOT/'validation_v11_predictions.csv').exists():
 old=pd.read_csv(ROOT/'validation_v11_predictions.csv');assert np.max(np.abs(oof.v11_same_splits-old.oof_prediction))<1e-8
rng=np.random.default_rng(18071);idx=np.stack(list(train.groupby('mission_id',sort=False).indices.values()));draw=np.array([rng.choice(len(idx),50,replace=False) for _ in range(20000)]);old_error=(oof.v11_same_splits-oof.actual).to_numpy()**2;new_error=(oof.oof_prediction-oof.actual).to_numpy()**2;gain=100/(1+np.sqrt(new_error[idx].mean(1)[draw].mean(1)))-100/(1+np.sqrt(old_error[idx].mean(1)[draw].mean(1)))
audit={'passed':True,'rows':len(csv),'score_recomputed':score,'v11_same_splits':report['v11_same_splits'],'local_gain':score-report['v11_same_splits']['score'],'source_and_data_hashes_verified':True,'public_score':None,'public_sized_validation_gain_quantiles':np.quantile(gain,[.05,.5,.95]).tolist(),'public_sized_validation_fraction_improved':float(np.mean(gain>0))}
(ROOT/'final_audit_v12.json').write_text(json.dumps(audit,indent=2));print(audit)
