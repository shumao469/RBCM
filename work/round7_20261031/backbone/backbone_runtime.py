"""Pinned frozen model adapters. No labels or outcome-driven model selection."""
from __future__ import annotations
import os,sys,json,hashlib,types,importlib.util
from pathlib import Path
TASK=Path('/mnt/h/2026/RBCM_round7_20261031')
for key in ('TMPDIR','TEMP','TMP'): os.environ[key]=str(TASK/'tmp')
for key in ('TORCH_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH'):
    path=TASK/'runtime_cache'/key;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for p in (TASK/'tmp',TASK/'runtime_cache'):p.mkdir(parents=True,exist_ok=True)
import numpy as np
import cv2
import torch
from PIL import Image
torch.set_num_threads(2);cv2.setNumThreads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.set_float32_matmul_precision('highest')
try:torch.set_num_interop_threads(1)
except RuntimeError:pass
WORK=Path(__file__).resolve().parents[2]
ROOT=WORK.parent
REPO=WORK/'github_release_snapshot'
WEIGHTS=Path('/mnt/h/2026/RBCM_review_20261028/asset_staging/pretrained_5afdbfcd066f/RBCM-Edge-Pretrained/pretrained')
TEED=TASK/'tools'/'teed_official'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

class FrozenBackbone:
    def __init__(self,name,device='cpu'):
        self.name=name;self.device=torch.device(device)
        if name=='teed_biped':
            # ted imports only count_parameters from a heavyweight visualization
            # module. Supply its exact pure counter; never change model code.
            stub=types.ModuleType('utils.img_processing')
            stub.count_parameters=lambda model:sum(p.numel() for p in model.parameters() if p.requires_grad)
            sys.modules['utils.img_processing']=stub
            sys.path.insert(0,str(TEED))
            from ted import TED
            self.checkpoint=TEED/'checkpoints/BIPED/5/5_model.pth'
            self.model=TED();self.model.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=True),strict=True)
            self.config={'official_repository':'https://github.com/xavysp/TEED','commit':'40fa4b1391dc6424f88989d0ca75d5b592c8681d','checkpoint_rule':'official main.py default 5/5_model.pth; no checkpoint search','training_domain':'official BIPED checkpoint; exact training IDs not reconstructed; RBCM source-val may overlap TEED training','input':'OpenCV BGR float32 0..255 minus source-fixed [103.939,116.779,123.68]; if either dimension<512 enlarge 1.5, then resize to next multiple of8 as official TestDataset','output':'sigmoid(last fused logits), bilinear restored native size; retain unquantized raw probability, no author visualization minmax/inversion','mean_difference':'Use the source BIPED mean on every domain, not dataset-specific target mean','import_compatibility':'Only unused visualization count_parameters import replaced by exact pure counter; architecture and checkpoint untouched'}
        else:
            source={'edgeanchor_biped':'biped','edgeanchor_multicue':'multicue'}[name]
            identity=load_json(Path('/mnt/h/2026/RBCM_review_20261028/source_cache')/source/'identity.json')
            self.config=identity['evaluation_config'];self.input_size=int(self.config['dataset']['input_size'])
            sys.path[:0]=[str(REPO),str(REPO/'src')]
            from edge_model.models.build import build_model
            from edge_model.data.transforms import make_eval_transform
            self.checkpoint=WEIGHTS/({'biped':'biped/main','multicue':'multicue_strict'}[source])/'best.pt'
            checkpoint=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
            self.model=build_model(self.config);self.model.load_state_dict(checkpoint['model'],strict=True)
            self.transform=make_eval_transform(self.input_size,preserve_aspect=True,native_size=False,binarize_edges=False,gt_mode='soft',return_meta=True)
        self.model.requires_grad_(False);self.model.eval();self.model.to(self.device)
        self.checkpoint_sha256=sha(self.checkpoint)
        self.parameters=sum(p.numel() for p in self.model.parameters())
    def prepare(self,rgb):
        h,w=rgb.shape[:2]
        if self.name=='teed_biped':
            bgr=rgb[:,:,::-1].copy()
            if h<512 or w<512:bgr=cv2.resize(bgr,(0,0),fx=1.5,fy=1.5)
            nh,nw=bgr.shape[:2]
            if nh%8 or nw%8:bgr=cv2.resize(bgr,(((nw//8)+1)*8,((nh//8)+1)*8))
            array=bgr.astype(np.float32)-np.array([103.939,116.779,123.68],np.float32)
            tensor=torch.from_numpy(array.transpose(2,0,1).copy())[None]
            return tensor,{'original_height':h,'original_width':w,'network_shape':list(tensor.shape)}
        dummy=Image.new('L',(w,h),0)
        tensor,_,meta=self.transform(Image.fromarray(rgb),dummy)
        return tensor[None],{**meta,'original_height':h,'original_width':w,'network_shape':[1,*tensor.shape]}
    def logits(self,tensor):
        outputs=self.model(tensor)
        return outputs[-1] if self.name=='teed_biped' else outputs['logits']
    def restore(self,prob,meta):
        if self.name!='teed_biped':
            x,y=int(meta['pad_left']),int(meta['pad_top']);cw,ch=int(meta['content_width']),int(meta['content_height'])
            prob=prob[y:y+ch,x:x+cw]
            # Same PIL floating-point bilinear restoration used by release.
            return np.asarray(Image.fromarray(prob.astype(np.float32)).resize((meta['original_width'],meta['original_height']),Image.Resampling.BILINEAR),np.float32)
        return cv2.resize(prob,(meta['original_width'],meta['original_height']),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    @torch.inference_mode()
    def infer_rgb(self,rgb):
        tensor,meta=self.prepare(rgb)
        prob=torch.sigmoid(self.logits(tensor.to(self.device)))[0,0].float().cpu().numpy()
        return self.restore(prob,meta),meta
    def manifest(self):
        return {'backbone_id':self.name,'checkpoint':str(self.checkpoint),'checkpoint_sha256':self.checkpoint_sha256,'parameters':self.parameters,'config':self.config,'adapter_sha256':sha(__file__),'frozen':True}

def read_rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB'),np.uint8)

def sobel_from_rgb(rgb):
    # Exact release edge_energy_from_image uses PIL RGB->L and float Sobel.
    gray=np.asarray(Image.fromarray(rgb).convert('L'),np.float32)/255.
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(gray,1,mode='reflect'),(3,3))
    kx=np.asarray([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],np.float32)
    gx=np.einsum('ijkl,kl->ij',windows,kx,optimize=True);gy=np.einsum('ijkl,kl->ij',windows,kx.T,optimize=True)
    energy=np.sqrt(gx*gx+gy*gy+1e-8);mean=float(energy.mean());std=float(energy.std())
    return (np.clip(energy/(mean+2*std+1e-6),0,3)/3).astype(np.float32)
