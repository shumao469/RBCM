"""Freeze HED+RBCM arm settings using only the historical BIPED source development."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import importlib.util
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
OLD_CODE = ROOT / 'work/cid_ap_hed_20261109/hed/run.py'
OLD_OUT = ROOT / 'outputs/RBCM_CID_AP_HED_20261109/hed'
OLD_CACHE = Path('/mnt/h/2026/RBCM_CID_AP_HED_20261109/hed/counts/source_biped')
RELEASE = ROOT / 'outputs/RBCM_round3_20261029/engineering/frozen_release_parameters.json'
OUT = ROOT / 'outputs/RBCM_HED_AUG_20261110/source'
CACHE = Path('/mnt/h/2026/RBCM_HED_AUG_20261110/source')

spec = importlib.util.spec_from_file_location('prior_hed_cid_runner', OLD_CODE)
hed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hed)
from ablation_operator import apply_prepared, build_candidate_configs, prepared
import ablation_operator
from backbone_runtime import read_rgb, sobel_from_rgb
import backbone_runtime
from run import score
import numpy as np
from PIL import Image

GRID = hed.SOURCE_THRESHOLDS
BASE = 'calibrated_base'
PLAIN = 'plain_identity'
LEGACY = 'legacy_main'
GAUSSIAN = 'gaussian_equal_context'
FAMILIES = ('equal_context', 'smoothing_equal_context', 'pa_context_only', 'sobel_context_only', GAUSSIAN)
OPTIONS = {family: [BASE, f'{family}_m0p5', f'{family}_m1p0'] for family in FAMILIES}
OPTIONS.update({name: [name] for name in (PLAIN, BASE, LEGACY)})
MEASURED = tuple(sorted({name for choices in OPTIONS.values() for name in choices} - {PLAIN}))
NPZ_KEYS = ('prob', 'edge')


def status(state='running', **details):
    result = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **details)
    hed.save(OUT / 'status.json', result)
    print(json.dumps(result), flush=True)


def receipt_path(sid, candidate=None):
    if candidate is None:
        return CACHE / 'maps' / f'{sid}.json'
    return CACHE / 'counts' / sid / f'{candidate}.json'


def verify_inputs():
    previous = hed.read(OLD_OUT / 'protocol.json')
    previous_sha = hed.digest(OLD_OUT / 'protocol.json')
    config = build_candidate_configs(hed.read(RELEASE)['biped']['candidates']['main_surround'], 'biped', 'hed_author_bsds')
    for suffix in ('_m0p5', '_m1p0'):
        config[GAUSSIAN + suffix] = dict(config['equal_context' + suffix], family=GAUSSIAN,
                                          context_operator='matched_gaussian')
    configs = {name: config[name] for name in (PLAIN, *MEASURED)}
    provenance = previous['provenance']
    if (hed.digest(hed.WEIGHTS) != provenance['weights']['sha256']
            or hed.digest(hed.COMPAT_DEPLOY) != provenance['compatible_deploy_sha256']):
        raise ValueError('Previous HED author graph/weight identity changed')
    if len(previous['source_entries']) != 20 or previous['source_threshold_grid'] != GRID.tolist():
        raise ValueError('Previous HED source scope or threshold grid changed')
    plain_freeze = hed.read(OLD_OUT / 'source_frozen.json')
    if plain_freeze['protocol_sha256'] != previous_sha or plain_freeze['source'] != 'BIPED development only':
        raise ValueError('Previous plain HED threshold not linked to source protocol')
    entries = []
    for original in previous['source_entries']:
        row = dict(original)
        sid = row['sample_id']
        plain = OLD_CACHE / f'{sid}.json'
        record = hed.read(plain)
        if (record['sample_id'] != sid or record['binding_sha256'] != previous_sha
                or record['image_sha256'] != row['image_sha256'] or record['gt_sha256'] != row['gt_sha256']
                or record['thresholds'] != GRID.tolist()):
            raise ValueError(f'HED source receipt identity changed: {sid}')
        row['plain_counts_sha256'] = hed.digest(plain)
        entries.append(row)
    if len({row['sample_id'] for row in entries}) != 20:
        raise ValueError('HED source contains duplicate IDs')
    protocol = dict(
        source='BIPED historical 20-image development partition', source_entries=entries,
        target_score_accessed_for_selection=False, source_threshold_grid=GRID.tolist(),
        selection='Max pooled F; tie -> smaller m then lower threshold; identical grids by family',
        prior_plain_hed_threshold=plain_freeze['threshold'], prior_hed_protocol_sha256=previous_sha,
        prior_hed_frozen_sha256=hed.digest(OLD_OUT / 'source_frozen.json'), hed_provenance=provenance,
        parameter_origin='BIPED frozen release parameters, matching the prior TEED transfer; m and threshold only',
        release_config_sha256=hed.digest(RELEASE), operator_sha256=hed.digest(ablation_operator.__file__),
        previous_hed_runner_sha256=hed.digest(OLD_CODE), sobel_runtime_sha256=hed.digest(backbone_runtime.__file__),
        matched_gaussian_and_match_runner_sha256=hed.digest(sys.modules['run'].__file__),
        evaluator=dict(matcher_sha256=previous['matcher']['unified_matcher_sha256'], nms_low=.02, max_dist=.0075,
                       binary_thinning=True), runner_sha256=hed.digest(__file__),
        options=OPTIONS, configurations=configs,
    )
    return protocol, configs, entries


def maps(entry, binding):
    sid = entry['sample_id']
    receipt = receipt_path(sid)
    raw = receipt.with_suffix('.npz')
    if receipt.exists():
        record = hed.read(receipt)
        if (record['binding_sha256'] != binding or record['sample_id'] != sid or hed.digest(raw) != record['map_sha256']):
            raise ValueError(f'Source HED map checkpoint verification failed: {sid}')
        return
    started = time.monotonic()
    if hed.digest(entry['image']) != entry['image_sha256']:
        raise ValueError(f'Source image changed: {sid}')
    prob = hed.infer(entry['image'])
    edge = sobel_from_rgb(read_rgb(entry['image']))
    if prob.shape != edge.shape:
        raise ValueError(f'Source HED and Sobel image size differs: {sid}')
    binary = io.BytesIO()
    np.savez_compressed(binary, prob=prob, edge=edge)
    hed.atomic(raw, binary.getvalue())
    hed.save(receipt, dict(sample_id=sid, binding_sha256=binding, map_sha256=hed.digest(raw),
                           shape=list(prob.shape), worker_seconds=time.monotonic() - started))


def fields_and_gt(entry, binding, base):
    sid = entry['sample_id']
    receipt = receipt_path(sid)
    raw = receipt.with_suffix('.npz')
    record = hed.read(receipt)
    if record['binding_sha256'] != binding or record['sample_id'] != sid or hed.digest(raw) != record['map_sha256']:
        raise ValueError(f'Cached BIPED map verification failed: {sid}')
    with np.load(raw, allow_pickle=False) as stored:
        prob, edge = (stored[key] for key in NPZ_KEYS)
    if hed.digest(entry['gt']) != entry['gt_sha256']:
        raise ValueError(f'Source GT changed: {sid}')
    with Image.open(entry['gt']) as image:
        gt = np.asarray(image.convert('L')) > 0
    if gt.shape != prob.shape:
        raise ValueError(f'Source GT shape differs: {sid}')
    return prepared(prob, edge, base), gt


def scores(entry, binding, configs):
    sid = entry['sample_id']
    fields, gt = fields_and_gt(entry, binding, configs[BASE])
    for name in MEASURED:
        path = receipt_path(sid, name)
        if path.exists():
            valid_counts(path, binding, sid, name)
            continue
        started = time.monotonic()
        counts = hed.match(score(fields, configs[name]), [gt], GRID, .0075)
        hed.save(path, dict(sample_id=sid, candidate=name, binding_sha256=binding, thresholds=GRID.tolist(),
                            counts=counts, worker_seconds=time.monotonic() - started))


def valid_counts(path, binding, sid, name):
    row = hed.read(path)
    if (row['binding_sha256'] != binding or row['sample_id'] != sid or row.get('candidate', PLAIN) != name
            or row['thresholds'] != GRID.tolist()):
        raise ValueError(f'Count receipt identity changed: {path}')
    counts = np.asarray(row['counts'])
    if counts.shape != (len(GRID), 4) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError(f'Invalid source counts: {path}')
    return counts


def count_progress(phase, entries):
    if phase == 'biped_hed_maps':
        return sum(receipt_path(entry['sample_id']).exists() for entry in entries)
    return sum(receipt_path(entry['sample_id'], name).exists() for entry in entries for name in MEASURED)


def run_phase(phase, worker, entries, workers, *args):
    prior = count_progress(phase, entries)
    inference = phase == 'biped_hed_maps'
    stage_total = len(entries) if inference else len(entries) * len(MEASURED)
    total = len(entries) * (1 + len(MEASURED))
    previous = 0 if inference else len(entries)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        pending = {pool.submit(worker, entry, *args) for entry in entries}
        last = None
        while pending:
            complete = count_progress(phase, entries)
            fresh = complete - prior
            if complete != last:
                eta = round((time.monotonic() - started) * (stage_total - complete) / fresh) if fresh >= min(workers, stage_total - prior) and fresh else None
                status(phase=phase, completed=previous + complete, total=total, phase_completed=complete,
                       phase_total=stage_total, eta_seconds=eta, eta_scope='current phase', workers=workers)
                last = complete
            finished, pending = wait(pending, timeout=8, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
    status(phase=phase, completed=previous + stage_total, total=total, phase_completed=stage_total,
           phase_total=stage_total, eta_seconds=0, workers=workers)


def choose(entries, protocol, binding, configs):
    sums = {name: np.zeros((len(GRID), 4), dtype=np.float64) for name in configs}
    for entry in entries:
        sid = entry['sample_id']
        old = OLD_CACHE / f'{sid}.json'
        if hed.digest(old) != entry['plain_counts_sha256']:
            raise ValueError(f'Plain HED source counts changed: {sid}')
        sums[PLAIN] += valid_counts(old, protocol['prior_hed_protocol_sha256'], sid, PLAIN)
        for name in MEASURED:
            sums[name] += valid_counts(receipt_path(sid, name), binding, sid, name)
    f = {name: hed.fscore(counts) for name, counts in sums.items()}
    selected = {}
    for family, choices in OPTIONS.items():
        ranked = [(float(f[name][index]), -configs[name]['multiplier'], -float(tau), name)
                  for name in choices for index, tau in enumerate(GRID)]
        value, _, negative_tau, name = max(ranked)
        tau = -negative_tau
        selected[family] = dict(config=configs[name], threshold=tau, selected_name=name,
                                F_source_development=value, threshold_on_grid_boundary=tau in (GRID[0], GRID[-1]))
    if selected[PLAIN]['threshold'] != protocol['prior_plain_hed_threshold']:
        raise ValueError('Source-only plain HED selection differs from previous freeze')
    return dict(source=protocol['source'], selected=selected, source_n_images=len(entries),
                protocol_sha256=binding, target_score_accessed_for_selection=False,
                source_F_by_threshold={key: values.tolist() for key, values in f.items()})


def pilot(entry, configs, binding):
    sid = entry['sample_id']
    start = time.monotonic()
    maps(entry, binding)
    elapsed_map = time.monotonic() - start
    fields, gt = fields_and_gt(entry, binding, configs[BASE])
    plain = hed.read(OLD_CACHE / f'{sid}.json')
    start = time.monotonic()
    actual = hed.match(fields['prob'], [gt], GRID[:1], .0075)[0]
    if actual != plain['counts'][0]:
        raise ValueError('Author HED prior source count parity failed at τ=.38')
    elapsed_plain = time.monotonic() - start
    start = time.monotonic()
    counts = hed.match(apply_prepared(fields, configs[BASE]), [gt], GRID, .0075)
    elapsed_base = time.monotonic() - start
    hed.save(receipt_path(sid, BASE), dict(sample_id=sid, candidate=BASE, binding_sha256=binding,
                                          thresholds=GRID.tolist(), counts=counts, worker_seconds=elapsed_base))
    report = dict(sample_id=sid, binding_sha256=binding, plain_counts_exact=True,
                  map_seconds=elapsed_map, plain_one_threshold_seconds=elapsed_plain,
                  base_six_thresholds_seconds=elapsed_base, source_only=True)
    hed.save(OUT / 'pilot.json', report)
    print(json.dumps(report), flush=True)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, configs, entries = verify_inputs()
        binding = hed.freeze(OUT / 'protocol.json', protocol)
        if args.pilot:
            pilot(entries[0], configs, binding)
            return
        pilot_result = hed.read(OUT / 'pilot.json')
        if pilot_result['binding_sha256'] != binding or not pilot_result['plain_counts_exact']:
            raise ValueError('Source parity pilot required before running the full study')
        run_phase('biped_hed_maps', maps, entries, args.inference_workers, binding)
        run_phase('source_six_threshold_matching', scores, entries, args.workers, binding, configs)
        frozen = choose(entries, protocol, binding, configs)
        hed.freeze(OUT / 'source_frozen.json', frozen)
        total = len(entries) * (1 + len(MEASURED))
        status('complete', phase='source_frozen', completed=total, total=total, eta_seconds=0,
               frozen_path=str(OUT / 'source_frozen.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--inference-workers', type=int, default=6)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if not args.pilot and not isinstance(error, BlockingIOError):
            status('failed', phase='error', error=str(error), traceback=traceback.format_exc())
        raise
