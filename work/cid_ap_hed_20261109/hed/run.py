"""Evaluate the author-released HED fusion on the fixed BIPED→CID transfer."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'work/validation_20261108/engineering'
ADMISSION = ROOT / 'outputs/RBCM_validation_20261108/engineering/target_admission.json'
OUT = ROOT / 'outputs/RBCM_CID_AP_HED_20261109/hed'
CACHE = Path('/mnt/h/2026/RBCM_CID_AP_HED_20261109/hed')
ASSETS = CACHE / 'assets'
REVISION = '912632b986acc6dd6cc33b95603b2f279d7bd9f2'
ORIGINAL_DEPLOY = ASSETS / 'deploy.prototxt'
WEIGHTS = ASSETS / 'hed_pretrained_bsds.caffemodel'
WEIGHTS_SHA = '4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7'
DEPLOY_SHA = '378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a'
COMPAT_DEPLOY = ASSETS / 'deploy_opencv.prototxt'
MEAN_BGR = (104.00698793, 116.66876762, 122.67891434)
CROPS = (34, 35, 36, 38, 42)
NET = None

sys.path.insert(0, str(WORK.parent))
sys.path.insert(0, str(PREVIOUS))
from metrics import THRESHOLDS, source_threshold, summarize_counts
from run import CID, THRESHOLDS as SOURCE_THRESHOLDS, fscore, match, source_entries, target_labels
import cv2
import numpy as np
from PIL import Image


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.part')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


def save(path, document):
    atomic(path, json.dumps(document, indent=2, allow_nan=False).encode('utf-8'))


def freeze(path, document):
    if path.exists():
        if read(path) != document:
            raise ValueError(f'Resume protocol mismatch at {path}')
    else:
        save(path, document)
    return digest(path)


def state(status='running', **details):
    record = dict(status=status, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), **details)
    save(OUT / 'status.json', record)
    print(json.dumps(record), flush=True)


def verify_assets():
    if digest(ORIGINAL_DEPLOY) != DEPLOY_SHA or digest(WEIGHTS) != WEIGHTS_SHA:
        raise ValueError('Author HED graph/weights failed SHA-256 integrity verification')
    # HED's modified Caffe derives crop origins by spatial coordinate alignment.
    # A pad-35 3x3 conv shifts 34 samples. n pool/deconv stages add 2**(n-1).
    # Standard OpenCV Crop requires the same origins explicitly and unique names.
    deploy = ORIGINAL_DEPLOY.read_text(encoding='utf-8')
    for stage, offset in enumerate(CROPS, 1):
        pattern = rf"name: 'crop'(.*?top: 'upscore-dsn{stage}')\s*}}"
        replacement = rf"name: 'crop-dsn{stage}'\1 crop_param {{ axis: 2 offset: {offset} }} }}"
        deploy, count = re.subn(pattern, replacement, deploy)
        if count != 1:
            raise ValueError(f'Unexpected deploy crop layer at stage {stage}')
    generated = deploy.encode('utf-8')
    if not COMPAT_DEPLOY.exists() or COMPAT_DEPLOY.read_bytes() != generated:
        atomic(COMPAT_DEPLOY, generated)
    return dict(
        weights={'url': 'https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel',
                 'bytes': WEIGHTS.stat().st_size, 'sha256': WEIGHTS_SHA},
        author_repo=f'https://github.com/s9xie/hed/tree/{REVISION}',
        author_deploy=f'https://github.com/s9xie/hed/blob/{REVISION}/examples/hed/deploy.prototxt',
        author_tutorial=f'https://github.com/s9xie/hed/blob/{REVISION}/examples/hed/HED-tutorial.ipynb',
        author_crop=f'https://github.com/s9xie/hed/blob/{REVISION}/src/caffe/layers/crop_layer.cpp',
        citation='https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html',
        original_deploy_sha256=DEPLOY_SHA, compatible_deploy_sha256=digest(COMPAT_DEPLOY),
        inference='OpenCV DNN CPU, original native size, raw sigmoid-fuse, no pre-NMS output rescale',
        opencv_version=cv2.__version__, mean_bgr=list(MEAN_BGR), crop_offsets=list(CROPS),
        model_training='Author-released BSDS-trained HED, no fine-tuning, not an equal training-data comparison',
    )


def infer(path):
    global NET
    if NET is None:
        cv2.setNumThreads(1)
        NET = cv2.dnn.readNetFromCaffe(str(COMPAT_DEPLOY), str(WEIGHTS))
        NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    bgr -= np.asarray(MEAN_BGR, dtype=np.float32)
    NET.setInput(np.ascontiguousarray(bgr.transpose(2, 0, 1)[None]))
    raw = NET.forward('sigmoid-fuse')
    if raw.shape != (1, 1, *rgb.shape[:2]):
        raise ValueError(f'HED output shape {raw.shape} does not match native image {rgb.shape}')
    edge = raw[0, 0]
    if not np.isfinite(edge).all() or edge.min() < 0 or edge.max() > 1 or edge.min() == edge.max():
        raise ValueError(f'Invalid HED fused probabilities at {path}')
    return edge


def biped_entries():
    entries = []
    for row in source_entries('teed_biped'):
        cache = Path(row['cache_file'])
        prior = read(row['prior_counts'])
        if (prior['sample_id'] != row['sample_id'] or prior['prediction_cache_sha256'] != row['cache_sha256']
                or digest(cache) != row['cache_sha256']):
            raise ValueError(f'BIPED source checkpoint mismatch {row["sample_id"]}')
        with np.load(cache, allow_pickle=False) as stored:
            identity = json.loads(stored['metadata_json'].item())['identity']
        entries.append(dict(sample_id=row['sample_id'], image=row['image_path'], gt=row['gt_path'],
                            image_sha256=identity['image_sha256'], gt_sha256=prior['gt_sha256']))
    if len(entries) != 20 or len({row['sample_id'] for row in entries}) != 20:
        raise ValueError('Expected the historical 20-image BIPED source-development partition')
    return entries


def work_one(phase, entry, binding):
    sid = entry['sample_id']
    checkpoint = CACHE / 'counts' / phase / f'{sid}.json'
    thresholds = SOURCE_THRESHOLDS if phase == 'source_biped' else THRESHOLDS
    if checkpoint.exists():
        saved = read(checkpoint)
        if saved['binding_sha256'] != binding or saved['sample_id'] != sid or saved['thresholds'] != thresholds.tolist():
            raise ValueError(f'HED checkpoint protocol or identity mismatch at {checkpoint}')
        counts = np.asarray(saved['counts'])
        if counts.shape != (len(thresholds), 4) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f'Invalid HED checkpoint counts at {checkpoint}')
        return saved, True
    started = time.monotonic()
    if digest(entry['image']) != entry['image_sha256'] or digest(entry['gt']) != entry['gt_sha256']:
        raise ValueError(f'HED input receipt changed: {phase}/{sid}')
    edge = infer(entry['image'])
    inference_seconds = time.monotonic() - started
    if phase == 'source_biped':
        with Image.open(entry['gt']) as image:
            labels = [np.asarray(image.convert('L')) > 0]
    else:
        labels = target_labels(CID, entry['gt'])
    if labels[0].shape != edge.shape:
        raise ValueError(f'HED edge/label shapes do not match: {phase}/{sid}')
    counts = match(edge, labels, thresholds, .0075)
    record = dict(sample_id=sid, binding_sha256=binding, thresholds=thresholds.tolist(), counts=counts,
                  image_sha256=entry['image_sha256'], gt_sha256=entry['gt_sha256'],
                  inference_and_input_verification_seconds=inference_seconds,
                  worker_seconds=time.monotonic() - started)
    save(checkpoint, record)
    return record, False


def run_phase(pool, phase, entries, binding, workers):
    records = []
    fresh = 0
    total = len(entries)
    submitted = time.monotonic()
    phase_progress = 0 if phase == 'source_biped' else 20
    cached = sum((CACHE / 'counts' / phase / f'{row["sample_id"]}.json').exists() for row in entries)
    state(phase=phase, completed=phase_progress + cached, total=58, phase_completed=cached, phase_total=total,
          eta_seconds=None, workers=workers, source_threshold_locked=(phase != 'source_biped'))
    tasks = [pool.submit(work_one, phase, row, binding) for row in entries]
    for task in as_completed(tasks):
        row, reused = task.result()
        records.append(row)
        if reused:
            continue
        fresh += 1
        complete = min(cached + fresh, total)
        elapsed = time.monotonic() - submitted
        eta = round(elapsed * (total - complete) / fresh) if fresh >= min(workers, total - cached) else None
        state(phase=phase, completed=phase_progress + complete, total=58, phase_completed=complete, phase_total=total,
              eta_seconds=eta, eta_scope='current phase', fresh_receipts=fresh, workers=workers,
              source_threshold_locked=(phase != 'source_biped'))
    return sorted(records, key=lambda row: row['sample_id'])


def main(args):
    provenance = verify_assets()
    if args.verify_adapter:
        entries = biped_entries()
        started = time.monotonic()
        a = infer(entries[0]['image'])
        elapsed = time.monotonic() - started
        # Exercise odd native shapes and the same loaded net's dynamic reconfiguration.
        odd = WORK / 'pilot_input.ppm'
        try:
            with Image.open(entries[0]['image']) as image:
                image.crop((0, 0, 319, 241)).save(odd)
            b = infer(odd)
            repeated = infer(entries[0]['image'])
            if not np.array_equal(a, repeated):
                raise ValueError('Changing native input size affected repeated HED result')
        finally:
            odd.unlink(missing_ok=True)
        result = dict(provenance=provenance, shapes=[list(a.shape), list(b.shape)],
                      native_repeat_exact=True, first_inference_seconds=elapsed, sample_id=entries[0]['sample_id'])
        save(OUT / 'adapter_pilot.json', result)
        print(json.dumps(result))
        return

    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state(phase='verify_inputs', completed=0, total=58, eta_seconds=None)
        source = biped_entries()
        protocol = dict(
            version=1, provenance=provenance, source_entries=source, source_name='BIPED development (Round 7)',
            source_threshold_grid=SOURCE_THRESHOLDS.tolist(), source_rule='pooled F max, lower threshold breaks ties',
            target_threshold_grid=THRESHOLDS.tolist(), target_name=CID, target_admission_sha256=digest(ADMISSION),
            matcher={'previous_runner_sha256': digest(PREVIOUS / 'run.py'),
                     'unified_matcher_sha256': digest(ROOT / 'work/round7_20261031/unified_bsds_evaluator.py'),
                     'low_threshold_nms': .02, 'max_dist_diagonal_fraction': .0075, 'thinning': True},
            runner_sha256=digest(__file__), metrics_sha256=digest(WORK.parent / 'metrics.py'),
        )
        protocol_sha = freeze(OUT / 'protocol.json', protocol)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            source_records = run_phase(pool, 'source_biped', source, protocol_sha, args.workers)
            source_counts = np.asarray([row['counts'] for row in source_records])
            source_f = fscore(source_counts.sum(axis=0))
            tau = source_threshold(source_counts, SOURCE_THRESHOLDS)
            source_freeze = dict(source='BIPED development only', n_images=len(source_records),
                                 threshold=tau, threshold_on_grid_boundary=tau in (SOURCE_THRESHOLDS[0], SOURCE_THRESHOLDS[-1]),
                                 grid=SOURCE_THRESHOLDS.tolist(), source_F=source_f.tolist(),
                                 protocol_sha256=protocol_sha, target_score_accessed_for_selection=False)
            frozen_sha = freeze(OUT / 'source_frozen.json', source_freeze)
            target = read(ADMISSION)['datasets'][CID]['entries']
            if len(target) != 38 or len({row['sample_id'] for row in target}) != 38:
                raise ValueError('CID must contain the 38 previously admitted images')
            records = run_phase(pool, 'target_cid', target, frozen_sha, args.workers)
        report = summarize_counts([row['counts'] for row in records], THRESHOLDS, tau)
        result = dict(backbone='hed_author_bsds', dataset=CID, arm='raw sigmoid-fuse', metrics=report,
                      source=source_freeze, frozen_source_sha256=frozen_sha,
                      target_sample_ids=[row['sample_id'] for row in records], n_images=len(records),
                      protocol_sha256=protocol_sha, notes='ODS and OIS are descriptive target oracles; deployed F uses BIPED source threshold.')
        save(OUT / 'results.json', result)
        state('complete', phase='summarized', completed=58, total=58, eta_seconds=0,
              source_threshold=tau, results_path=str(OUT / 'results.json'))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-adapter', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if not args.verify_adapter and not isinstance(error, BlockingIOError):
            state('failed', phase='error', error=str(error), traceback=traceback.format_exc())
        raise
