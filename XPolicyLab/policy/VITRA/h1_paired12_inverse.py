"""Explicit run-bound Wuji20 -> real H1 12D conditional decoder.

This estimates discarded joints from paired data; it is NOT an exact inverse.
No fallback, clipping, mimic packing, or change to the historical adapter.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''): digest.update(block)
    return digest.hexdigest()


class Paired12Decoder:
    def __init__(self,path,expected_sha,expected_provenance):
        path=Path(path)
        if not expected_sha or file_sha(path)!=expected_sha:
            raise ValueError('paired12 sidecar SHA mismatch')
        self.meta=json.loads(path.read_text())
        if self.meta['schema']!='vitra_0902_wuji20_h1_independent12_v1':
            raise ValueError('unsupported paired12 schema')
        for key,value in expected_provenance.items():
            # Mount aliases must not invalidate the same hash-bound checkpoint.
            if key=='checkpoint_weights_path' and self.meta['provenance'].get(key):
                if Path(self.meta['provenance'][key]).resolve()==Path(value).resolve():
                    continue
            if self.meta['provenance'].get(key)!=value:
                raise ValueError(f'paired12 provenance mismatch: {key}')
        for source in self.meta['sources']:
            if file_sha(source['path'])!=source['sha256']:
                raise ValueError(f"paired12 source SHA mismatch: {source['path']}")
        self.forests={}
        for side,entry in self.meta['forests'].items():
            forest=path.parent/entry['file']
            if file_sha(forest)!=entry['sha256']:
                raise ValueError(f'{side} paired12 forest SHA mismatch')
            with np.load(forest,allow_pickle=False) as data:
                self.forests[side]={key:data[key].copy() for key in data.files}
        self.lower=np.asarray(self.meta['lower_rad'])
        self.upper=np.asarray(self.meta['upper_rad'])

    def predict(self,side,q):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(20,) or not np.isfinite(q).all():
            raise ValueError(f'{side} paired12 expects finite (20,), got {q.shape}')
        forest=self.forests[side]
        outputs=[]
        # sklearn trees evaluate features as float32; match its trained split semantics.
        x=q.astype(np.float32)
        for start in forest['roots']:
            node=int(start)
            while forest['left'][node]>=0:
                feature=int(forest['feature'][node])
                node=int(forest['left'][node] if x[feature]<=forest['threshold'][node]
                         else forest['right'][node])
            outputs.append(forest['value'][node])
        result=np.mean(outputs,axis=0)
        if result.shape!=(12,) or not np.isfinite(result).all():
            raise ValueError('paired12 invalid output')
        if np.any(result<self.lower) or np.any(result>self.upper):
            raise ValueError(f'{side} paired12 output outside real H1 limits')
        return result


class Paired12Adapter:
    """Opt-in composition preserving observation conversion and state bookkeeping."""
    def __init__(self,base,decoder):
        self.base=base
        self.decoder=decoder

    def __getattr__(self,key):
        return getattr(self.base,key)

    def wuji_to_h1(self,q20,side,env_idx=0):
        q=np.asarray(q20,dtype=np.float64)
        if q.ndim==2:
            if q.shape[1]!=20 or not len(q):
                raise ValueError('paired12 batch must have shape (N,20), N>0')
            return np.stack([self.wuji_to_h1(row,side,env_idx) for row in q])
        q=self.base._validate_q20(side,q)
        result=self.decoder.predict(side,q)
        # Retain the existing 80 mm morphology check; do not confuse it with
        # correctness of independent12 recovery (measured separately offline).
        points=np.asarray(self.base.wuji_models[side].forward_points(q,canonical=True))
        points[:,0,:]=0  # same already-established landmark schema as base adapter
        inspire=self.base.inspire_models[side]
        q6=result[[8,0,2,6,4,9]]
        residual=inspire.forward_points(q6,canonical=True)-inspire.scale_target(points)
        distances=np.linalg.norm(residual,axis=-1)
        rms=float(np.sqrt(np.mean(distances**2)))
        diagnostic={'backend':'paired12','conditional_not_exact_inverse':True,
                    'rms_m':rms,'max_error_m':float(distances.max()),
                    'limit_m':self.base.inverse_max_rms_m,'output_q12':result.tolist(),
                    'quality_gate':{'accepted':bool(np.isfinite(rms) and rms<=self.base.inverse_max_rms_m)}}
        self.base.last_diagnostics[f'{env_idx}:{side}:wuji_to_h1']=diagnostic
        if not diagnostic['quality_gate']['accepted']:
            raise ValueError(f'{side} paired12 geometry quality gate failed: {diagnostic}')
        self.base._previous_inspire.setdefault(int(env_idx),{})[side]=q6.copy()
        return result.astype(np.float32)
