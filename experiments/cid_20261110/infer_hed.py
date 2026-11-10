"""Export source-locked HED/RBCM boundary probability maps for one image."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'work/round7_20261031'),
                str(ROOT / 'work/validation_20261108/engineering')]
from ablation_operator import prepared, sobel_energy_from_image
from run import score


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'SHA-256 mismatch for {path}')


def load_hed(deploy, weights, provenance):
    verify(deploy, provenance['original_deploy_sha256'])
    verify(weights, provenance['weights']['sha256'])
    graph = deploy.read_text(encoding='utf-8')
    for stage, offset in enumerate(provenance['crop_offsets'], 1):
        pattern = rf"name: 'crop'(.*?top: 'upscore-dsn{stage}')\s*}}"
        replacement = rf"name: 'crop-dsn{stage}'\1 crop_param {{ axis: 2 offset: {offset} }} }}"
        graph, count = re.subn(pattern, replacement, graph)
        if count != 1:
            raise ValueError(f'Unexpected crop layer at stage {stage}')
    graph_bytes = graph.encode('utf-8')
    if hashlib.sha256(graph_bytes).hexdigest() != provenance['compatible_deploy_sha256']:
        raise ValueError('Generated OpenCV deploy graph differs from the evaluated graph')
    cv2.setNumThreads(1)
    net = cv2.dnn.readNetFromCaffe(np.frombuffer(graph_bytes, dtype=np.uint8),
                                   np.fromfile(weights, dtype=np.uint8))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--deploy', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output .npz file')
    parser.add_argument('--protocol', choices=['allow_zero', 'nonzero'], default='allow_zero')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        parser.error('--output must end in .npz')

    provenance = json.loads((HERE / 'hed_provenance.json').read_text(encoding='utf-8'))
    frozen = json.loads((HERE / 'hed_source_locked.json').read_text(encoding='utf-8'))
    selected = frozen[args.protocol]['selected']
    net = load_hed(args.deploy, args.weights, provenance)
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    bgr -= np.asarray(provenance['mean_bgr'], dtype=np.float32)
    net.setInput(np.ascontiguousarray(bgr.transpose(2, 0, 1)[None]))
    raw = net.forward('sigmoid-fuse')
    if raw.shape != (1, 1, *rgb.shape[:2]) or not np.isfinite(raw).all():
        raise ValueError('Unexpected HED fused output')
    fields = prepared(raw[0, 0], sobel_energy_from_image(args.image), selected['calibrated_base']['config'])
    maps = {name: score(fields, item['config']) for name, item in selected.items()}
    np.savez_compressed(args.output, **maps)


if __name__ == '__main__':
    main()
