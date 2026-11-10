"""Frozen detector counterfactuals; synthetic images, no neural-response inference."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time,traceback
import numpy as np
from scipy.ndimage import gaussian_filter

WORK=Path(__file__).resolve().parents[1]; ROOT=WORK.parent
OUT=ROOT/'outputs/RBCM_validation_20261108/stimuli'; R7=ROOT/'outputs/RBCM_round7_20261031'
sys.path.insert(0,str(WORK/'round7_20261031/backbone'));sys.path.insert(0,str(WORK/'round7_20261031'))
from backbone_runtime import FrozenBackbone,sobel_from_rgb
import ablation_operator as op
MODELS=('teed_biped','edgeanchor_biped')
SEEDS=tuple(range(88001,88033));CS=(0,12,26);D=(-2,0,2);SS=('neutral','continued','reversed')
N=128; yy,xx=np.mgrid[:N,:N].astype(np.float32)-np.float32((N-1)/2)
rr=np.maximum(np.abs(yy),np.abs(xx));center=rr<=3.5; ann=(rr>3.5)&(rr<=14.5)
VARIANTS=('plain','calibrated_base','legacy_frozen','fusion_frozen','box_frozen','matched_sobel_ring','matched_sobel_box','matched_sobel_global','matched_sobel_ring_fixed_norm')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for i in range(100):
        try:os.replace(tmp,path);return
        except PermissionError:
            if i==99:raise
            time.sleep(.05)
def status(state,phase,done,eta=None,**extra):
    write(OUT/'status.json',{'status':state,'phase':phase,'completed':done,'total':len(SEEDS)*len(MODELS),'eta_seconds':eta,'updated_at_epoch':time.time(),**extra})

def stim(seed,c,d,context):
    rng=np.random.default_rng(seed);orient=seed%2;sgn=np.where((xx,yy)[orient]>0,1,-1)
    bg=gaussian_filter(rng.standard_normal((N,N)),sigma=2+(seed%5)/2)
    bg=np.rint(np.clip(bg/bg.std()*11,-36,36)).astype(np.int16)
    grey=(128+bg*np.clip((rr-14.5)/6,0,1)).round().astype(np.int16)
    env=np.minimum(np.clip((rr-3.5)/3,0,1),np.clip((15.5-rr)/3,0,1))
    a=np.rint((18+seed%19)*env).astype(np.int16)*sgn
    grey[ann]=128+{'neutral':0,'continued':1,'reversed':-1}[context]*a[ann]
    grey[center]=128+(c+d)*sgn[center]
    return grey.astype(np.uint8)

def raw_energy(gray):
    a=np.asarray(gray,np.float32)/255.;w=np.lib.stride_tricks.sliding_window_view(np.pad(a,1,mode='reflect'),(3,3))
    k=np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],np.float32)
    x=np.einsum('ijkl,kl->ij',w,k,optimize=True);y=np.einsum('ijkl,kl->ij',w,k.T,optimize=True)
    return np.sqrt(x*x+y*y+1e-8)
def denom(z):return float(z.mean())+2*float(z.std())+1e-6
def normalize(z,scale):return (np.clip(z/scale,0,3)/3).astype(np.float32)
def region(seed):return (rr<=2.5)&(np.abs((xx,yy)[seed%2])==.5)
def configs(model):
    p=(R7/'backbone/frozen_source/teed_biped.json') if model=='teed_biped' else (R7/'ablation/frozen_source/biped.json')
    f=read(p);c=f['selected_configs'];sel=f['selected_per_family'];base=c['legacy_main']
    variants={'plain':c['plain_identity'],'calibrated_base':c['calibrated_base'],'legacy_frozen':base,
      'fusion_frozen':c[sel['equal_context']],'box_frozen':c[sel['smoothing_equal_context']]}
    for name,pool in [('matched_sobel_ring','ring'),('matched_sobel_box','triple_box_smoothing'),('matched_sobel_global','global_sobel_reference'),('matched_sobel_ring_fixed_norm','ring')]:
        variants[name]={**base,'family':name,'multiplier':.5,'prob_weight':0.,'edge_weight':1.,'context_operator':pool}
    return variants,p,f['frozen_thresholds']

def one_seed(model,net,cfg,seed):
    roi=region(seed);scale=denom(raw_energy(stim(seed,0,0,'neutral')));records=[]
    for c in CS:
        for d in D:
            images=[stim(seed,c,d,s) for s in SS]
            assert all(np.array_equal(im[center],images[0][center]) for im in images)
            assert all(np.array_equal(im[~(center|ann)],images[0][~(center|ann)]) for im in images)
            assert np.array_equal(np.sort(images[1][ann]),np.sort(images[2][ann]))
            first=raw_energy(images[0])[roi]
            for context,im in zip(SS,images):
                e0=raw_energy(im);assert np.array_equal(e0[roi],first)
                rgb=np.repeat(im[:,:,None],3,axis=2);p,meta=net.infer_rgb(rgb)
                e=normalize(e0,denom(e0));fixed=normalize(e0,scale)
                if not records:assert np.allclose(sobel_from_rgb(rgb),e,atol=1e-7,rtol=1e-6)
                fields=op.prepared(p,e,cfg['legacy_frozen']);fields_fixed=op.prepared(p,fixed,cfg['legacy_frozen'])
                maps={n:float(op.apply_prepared(fields_fixed if n.endswith('fixed_norm') else fields,x)[roi].mean()) for n,x in cfg.items()}
                records.append({'model':model,'seed':seed,'center_contrast_u8':c,'probe_u8':d,'surround':context,
                    'center_sha256':hashlib.sha256(im[center].tobytes()).hexdigest(),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'mean_raw_sobel_roi':float(e0[roi].mean()),'mean_normalized_sobel_roi':float(e[roi].mean()),
                    'mean_reference_normalized_sobel_roi':float(fixed[roi].mean()),'mean_Pa_roi':float(p[roi].mean()),**maps})
    return records

def summarize(records):
    lookup={(r['model'],r['seed'],r['center_contrast_u8'],r['probe_u8'],r['surround']):r for r in records}
    rng=np.random.default_rng(881110);draw=rng.integers(0,len(SEEDS),size=(4000,len(SEEDS)));out=[]
    endpoints=[*VARIANTS,'mean_raw_sobel_roi','mean_normalized_sobel_roi','mean_reference_normalized_sobel_roi']
    for model in MODELS:
        for key in endpoints:
            def get(seed,c,d,s):return lookup[model,seed,c,d,s][key]
            def delta(seed,c):return get(seed,c,0,'continued')-get(seed,c,0,'reversed')
            def inc(seed,c):return delta(seed,c)-(lookup[model,seed,c,0,'continued']['calibrated_base']-lookup[model,seed,c,0,'reversed']['calibrated_base'])
            def probe(seed,c,s):return (get(seed,c,2,s)-get(seed,c,-2,s))/4
            for metric,f in [('fixed_center_continued_minus_reversed_c12',lambda seed:delta(seed,12)),
                ('difference_of_differences_c26_vs_c0',lambda seed:delta(seed,26)-delta(seed,0)),
                ('local_probe_slope_difference_c12',lambda seed:probe(seed,12,'continued')-probe(seed,12,'reversed')),
                *([('postprocess_increment_over_calibrated_c12',lambda seed:inc(seed,12))] if key in VARIANTS else [])]:
                a=np.array([f(s) for s in SEEDS]);lo,hi=np.quantile(a[draw].mean(1),[.025,.975]);out.append({'model':model,'readout':key,'metric':metric,'n_background_seeds':len(SEEDS),'difference':float(a.mean()),'low':float(lo),'high':float(hi)})
    return out
def csv_write(name,rows):
    path=OUT/name;tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def main(device):
    OUT.mkdir(parents=True,exist_ok=True);status('running','锁定刺激与冻结配置',0)
    allcfg={m:configs(m) for m in MODELS}
    protocol={'kind':'synthetic fixed-centre engineering stress test; no biology, no performance target, no selection',
      'models':list(MODELS),'seeds':list(SEEDS),'contrast_levels_u8':list(CS),'probe_signed_step_u8':list(D),'surrounds':list(SS),
      'center_lock_square_px':8,'probe_roi':'two edge-adjacent columns or rows within central 6 pixels; 3x3 Sobel support inside fixed centre',
      'annular_controls':'continued vs sign reversed, exact annular histogram and identical exterior; neutral contrast off, not histogram-matched',
      'endpoints':'mean pre-NMS probability, centre contrast interaction, local finite-difference probe slope and fixed E normalization diagnostic',
      'ci':'paired background-seed percentile intervals, 4000 draws, pointwise only',
      'configs':{m:{'source_file':str(v[1]),'sha256':sha(v[1]),'variants':v[0]} for m,v in allcfg.items()},'script_sha256':sha(__file__)}
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==protocol,'Frozen synthetic protocol changed; new output version required'
    else:write(p,protocol)
    signature=sha(p);completed=0;timed=[];records=[];model_meta={}
    for model in MODELS:
        net=FrozenBackbone(model,device);model_meta[model]=net.manifest();cfg=allcfg[model][0]
        for seed in SEEDS:
            record=OUT/'checkpoints'/f'{model}_{seed}.json'
            if record.exists():
                old=read(record);assert old['protocol_sha256']==signature and old['checkpoint_sha256']==net.checkpoint_sha256
                block=old['rows']
            else:
                t=time.monotonic();block=one_seed(model,net,cfg,seed);timed.append(time.monotonic()-t)
                write(record,{'protocol_sha256':signature,'checkpoint_sha256':net.checkpoint_sha256,'rows':block})
            records.extend(block);completed+=1;left=len(SEEDS)*len(MODELS)-completed
            status('running','同中心包围扰动：完整背景断点',completed,float(np.median(timed)*left) if timed else None,model=model,seed=seed,device=device)
        del net
    estimates=summarize(records);csv_write('paired_readouts.csv',records);csv_write('paired_effects.csv',estimates)
    write(OUT/'summary.json',{'status':'complete','protocol_sha256':signature,'models':model_meta,'n_conditions':len(records),'n_backgrounds':len(SEEDS),'estimates':estimates,'limitations':['Synthetic images; not physiological responses or natural target superiority.','No target fitting or threshold calibration; fixed m=.5 contrasts isolate pooling, not optimized methods.','Released normalized Sobel can vary at fixed local raw gradient because its scale uses whole-image energy.','Mean pre-NMS probability is a sensitivity diagnostic; not fixed-threshold F or shadow segmentation.']})
    status('complete','配对效应与区间已落盘',completed,0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
    try:main(a.device)
    except Exception as e:
        status('failed','任务失败',len(list((OUT/'checkpoints').glob('*.json'))),error=repr(e),traceback=traceback.format_exc());raise
