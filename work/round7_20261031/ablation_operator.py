"""Frozen-base information-ablation operators. NumPy/OpenCV; no model loading.

This module accepts a frozen backbone probability map Pa and an independently
image-derived Sobel map E. Image Pa is never presented as physical contrast.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
PREPROCESSING_VERSION = "release_PIL_L_3x3_Sobel_mu_plus_2sd_v1"
FAMILIES = ("pa_context_only", "sobel_context_only", "equal_context", "smoothing_equal_context", "pointwise_sobel_global_reference")
AMPLITUDES = (0., .5, 1.)


def sobel_energy_from_image(path):
    """Bitwise-equivalent formulation of the pinned release Sobel preprocessor."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), np.float32) / 255.
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(array, 1, mode="reflect"), (3, 3))
    kx = np.asarray([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], np.float32)
    gx = np.einsum("ijkl,kl->ij", windows, kx, optimize=True)
    gy = np.einsum("ijkl,kl->ij", windows, kx.T, optimize=True)
    energy = np.sqrt(gx*gx + gy*gy + 1.e-8)
    return np.clip(energy/(float(energy.mean())+2*float(energy.std())+1.e-6), 0, 3).astype(np.float32)/3


def blur(array, width):
    return cv2.blur(np.asarray(array, np.float32), (int(width), int(width)), borderType=cv2.BORDER_REFLECT_101)


def annulus(array, inner, outer):
    return (blur(array, outer)*(outer*outer)-blur(array, inner)*(inner*inner))/max(1., outer*outer-inner*inner)


def ring_surround(array, ring):
    i, o, j, k = (int(value) for value in ring.split("-"))
    return .62*annulus(array, i, o)+.38*annulus(array, j, k)


def smoothing_surround(array):
    near=blur(array,3); mid=blur(near,3); far=blur(mid,3)
    return .45*near+.35*mid+.20*far


def contrast(center, surround):
    return np.clip((center-surround)/(center+surround+.06),-1.,1.).astype(np.float32)


def sigmoid(value):
    return (1./(1.+np.exp(-np.clip(value,-30.,30.)))).astype(np.float32)


def prepared(prob, edge, config):
    p=np.asarray(prob,np.float32); e=np.asarray(edge,np.float32)
    if p.ndim!=2 or e.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(e).all():
        raise ValueError("Require finite same-shape 2D Pa and independently derived E")
    clipped=np.clip(p,1.e-4,1.-1.e-4)
    base=np.log(clipped/(1.-clipped)).astype(np.float32)*float(config["temperature"])+float(config["bias"])
    if config["sharpen"]:
        base=base+float(config["sharpen"])*(p-blur(p,3))
    uncertainty=np.clip(4.*p*(1.-p),0.,1.)
    if float(config["uncertainty_power"])!=1.:
        uncertainty=np.power(uncertainty,float(config["uncertainty_power"]))
    return {"prob":p,"edge":e,"base":base,"uncertainty":uncertainty}


def apply_prepared(fields, config):
    family=config["family"]
    if family=="plain_identity": return fields["prob"].copy()
    multiplier=float(config["multiplier"])
    if multiplier==0.: return sigmoid(fields["base"])
    context=config["context_operator"]
    if context=="ring":
        key="ring_state_"+config["ring"]
        if key not in fields:
            fields[key]=(contrast(fields["prob"],ring_surround(fields["prob"],config["ring"])),contrast(fields["edge"],ring_surround(fields["edge"],config["ring"])))
        cp,ce=fields[key]
    elif context=="triple_box_smoothing":
        key="smoothing_states"
        if key not in fields:
            fields[key]=(contrast(fields["prob"],smoothing_surround(fields["prob"])),contrast(fields["edge"],smoothing_surround(fields["edge"])))
        cp,ce=fields[key]
    elif context=="global_sobel_reference":
        key="global_sobel_state"
        if key not in fields: fields[key]=contrast(fields["edge"],float(fields["edge"].mean()))
        cp,ce=0.,fields[key]
    else: raise ValueError("Unknown context operator: "+context)
    state=np.clip(float(config["prob_weight"])*cp+float(config["edge_weight"])*ce,-1.,1.)
    return sigmoid(fields["base"]+float(config["alpha"])*multiplier*fields["uncertainty"]*state)


def apply(prob, edge, config):
    """Apply one fully specified source-frozen map transform; no NMS/threshold."""
    return apply_prepared(prepared(prob,edge,config),config)


def build_candidate_configs(release_main_config, source, backbone_id):
    common={key:release_main_config[key] for key in ("temperature","bias","sharpen","ring","uncertainty_power","alpha")}
    common.update({"schema_version":SCHEMA_VERSION,"source":source,"backbone_id":backbone_id,"preprocessing_version":PREPROCESSING_VERSION,"near_fraction":.62,"far_fraction":.38,"contrast_offset":.06,"historically_exposed_source_validation":True})
    configs={"calibrated_base":{**common,"family":"calibrated_base","context_operator":"ring","prob_weight":0.,"edge_weight":0.,"multiplier":0.}}
    definitions={"pa_context_only":("ring",1.,0.),"sobel_context_only":("ring",0.,1.),"equal_context":("ring",.5,.5),"smoothing_equal_context":("triple_box_smoothing",.5,.5),"pointwise_sobel_global_reference":("global_sobel_reference",0.,1.)}
    for family,(context,pw,ew) in definitions.items():
        for multiplier in (.5,1.):
            name=family+"_m"+str(multiplier).replace(".","p")
            configs[name]={**common,"family":family,"context_operator":context,"prob_weight":pw,"edge_weight":ew,"multiplier":multiplier}
    configs["legacy_main"]={**common,"family":"legacy_main","context_operator":"ring","prob_weight":release_main_config["prob_weight"],"edge_weight":release_main_config["edge_weight"],"multiplier":1.}
    configs["plain_identity"]={**common,"family":"plain_identity","context_operator":"ring","prob_weight":0.,"edge_weight":0.,"multiplier":0.}
    return configs


def family_candidate_names(configs, family):
    return ["calibrated_base"]+[name for name,value in configs.items() if value["family"]==family]
