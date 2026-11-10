"""Pinned original BSDS C++ one-to-one matcher with deterministic per-call seed.

NMS is a separate, common upstream step. Binary thinning and max_dist are
explicit. Counts retain all human annotators: precision matches are unioned
across annotators; recall sums each annotation separately.
"""
from pathlib import Path
import sys
import numpy as np

PORT=Path('/mnt/h/2026/RBCM_round7_20261031/tools/py-bsds500')
sys.path.insert(0,str(PORT))
from bsds.correspond_pixels import correspond_pixels,set_seed
from bsds.thin import binary_thin

SEED=4517

def evaluate_boundaries(predicted_boundaries,gt_boundaries,thresholds=99,
                        max_dist=.0075,apply_thinning=True,progress=None):
    # Preserve input values in double precision for MATLAB-compatible threshold
    # comparisons; casting PNG probabilities to float32 can move 85/255 across 1/3.
    prob=np.asarray(predicted_boundaries,dtype=np.float64)
    if prob.ndim!=2 or not np.isfinite(prob).all():
        raise ValueError('Expected finite HxW probability map')
    if prob.min()<0 or prob.max()>1:
        raise ValueError('Probability must be in [0,1]; no hidden rescaling')
    if isinstance(thresholds,(int,np.integer)):
        thresholds=np.linspace(1/(thresholds+1),1-1/(thresholds+1),thresholds)
    else:thresholds=np.asarray(thresholds,dtype=float)
    if thresholds.ndim!=1 or np.any(np.diff(thresholds)<0):
        raise ValueError('Thresholds must be a sorted one-dimensional array')
    gts=[np.asarray(g,dtype=bool) for g in gt_boundaries]
    if not gts or any(g.shape!=prob.shape for g in gts):
        raise ValueError('Nonempty GT list must match prediction dimensions')
    sr=np.full(len(thresholds),sum(int(g.sum()) for g in gts),dtype=float)
    cr=np.zeros(len(thresholds));cp=np.zeros(len(thresholds));sp=np.zeros(len(thresholds))
    # Identical masks at adjacent thresholds have identical deterministic counts.
    previous_key=None;previous_counts=None
    iterator=enumerate(thresholds)
    if progress is not None:iterator=progress(list(iterator))
    for i,t in iterator:
        binary=prob>=t
        key=np.packbits(binary,axis=None).tobytes()
        if key==previous_key:
            cr[i],cp[i],sp[i]=previous_counts
            continue
        if apply_thinning:binary=np.asarray(binary_thin(binary),dtype=bool)
        sp[i]=int(binary.sum())
        if sp[i]>0:
            acc=np.zeros(prob.shape,dtype=bool)
            for gt in gts:
                if not gt.any():continue
                set_seed(SEED)
                m1,m2,_,_=correspond_pixels(binary,gt,max_dist=max_dist,outlier_cost=100.)
                acc|=m1>0
                cr[i]+=int(np.count_nonzero(m2))
            cp[i]=int(acc.sum())
        previous_key=key;previous_counts=(cr[i],cp[i],sp[i])
    return cr,sr,cp,sp,thresholds

def summarize_counts(counts,thresholds):
    """Counts order is matchedPrediction,prediction,matchedTruth,truth.

    AP_BSDS follows collect_eval_bdry.m's 0:.01:1 interpolation with zero
    outside the observed recall range. AP_envelope_compat is an additional
    ranking diagnostic; do not silently interchange these two quantities.
    """
    counts=np.asarray(counts,dtype=float);thresholds=np.asarray(thresholds,float)
    p=counts[:,0]/np.maximum(counts[:,1],1)
    r=counts[:,2]/np.maximum(counts[:,3],1)
    f=2*p*r/np.maximum(p+r,1e-12)
    ru,idx=np.unique(r,return_index=True);pu=p[idx]
    ap=float(np.interp(np.arange(101)/100,ru,pu,left=0.,right=0.).sum()*.01) if len(ru)>1 else 0.
    mr=np.r_[0.,r,1.];mp=np.r_[0.,p,0.];order=np.argsort(mr,kind='stable');mr=mr[order];mp=mp[order]
    mp=np.maximum.accumulate(mp[::-1])[::-1]
    envelope=float(np.sum(np.diff(mr)*mp[1:]))
    i=int(np.argmax(f))
    return {'AP_BSDS_101point':ap,'AP_envelope_compat':envelope,
        'best_discrete_F':float(f[i]),'best_discrete_threshold':float(thresholds[i]),
        'precision':p.tolist(),'recall':r.tolist(),'F':f.tolist()}
