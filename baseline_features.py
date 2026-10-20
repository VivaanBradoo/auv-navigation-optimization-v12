"""Check and optionally regenerate the preserved v3/v4/v5 sensor features."""
import hashlib,json
from pathlib import Path
import pandas as pd
import navigation_v4 as old
import motion_v5,joint_v5
ROOT=Path(__file__).resolve().parent
SOURCES=['baseline_features.py','navigation_v3.py','navigation_v4.py','motion_v5.py','joint_v5.py','features.py','train.csv','test.csv','sample_submission.csv','train_timeseries.parquet','test_timeseries.parquet']
CACHES=['v3_sensor_features.pkl','v4_sensor_features.pkl','ar_motion_features.pkl','test_ar_motion_features.pkl','ar_joint_features.pkl','test_ar_joint_features.pkl']

def fingerprint():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}

def ensure(rebuild=False):
    hashes=fingerprint();manifest=ROOT/'baseline_v8_manifest.json'
    if not rebuild and all((ROOT/name).exists() for name in CACHES) and manifest.exists() and json.loads(manifest.read_text())==hashes:return
    tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv');r,p,c,rt,pt,ct,compass=old.original_features(tr,te,False)
    raw=pd.read_parquet(ROOT/'train_timeseries.parquet');rawt=pd.read_parquet(ROOT/'test_timeseries.parquet')
    m=old.extract_motion(tr,raw,r,p,compass);mt=old.extract_motion(te,rawt,rt,pt,compass);pd.to_pickle((m,mt),ROOT/'v4_sensor_features.pkl')
    for module,filename,prefix in [(motion_v5,'ar_motion_features.pkl','r90_'),(joint_v5,'ar_joint_features.pkl','a_')]:
        a=module.extract_motion(tr,raw,r,p,compass).filter(regex='^'+prefix);at=module.extract_motion(te,rawt,rt,pt,compass)
        a.to_pickle(ROOT/filename);at.to_pickle(ROOT/('test_'+filename))
    manifest.write_text(json.dumps(hashes,indent=2))
