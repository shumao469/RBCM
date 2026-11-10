"""Resume full-curve CID evaluation from frozen score maps and source settings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENGINEERING = ROOT / 'work/validation_20261108/engineering'
sys.path[:0] = [str(ENGINEERING), str(HERE.parent)]

# Import by a non-__main__ module name so spawned workers use the original
# matcher and the exact operator used by the original fixed-threshold sweep.
import run as prior

OUT = ROOT / 'outputs/RBCM_CID_AP_HED_20261109/ap'
THRESHOLDS = np.linspace(.01, .99, 99)
ARMS = (
    'plain_identity', 'equal_context', 'legacy_main', 'calibrated_base',
    'smoothing_equal_context', 'sobel_context_only', 'pa_context_only',
)
CONTRASTS = (
    ('equal_context', 'plain_identity'), ('legacy_main', 'plain_identity'),
    ('equal_context', 'calibrated_base'), ('legacy_main', 'calibrated_base'),
    ('equal_context', 'smoothing_equal_context'), ('legacy_main', 'smoothing_equal_context'),
    ('smoothing_equal_context', 'plain_identity'), ('sobel_context_only', 'plain_identity'),
    ('sobel_context_only', 'calibrated_base'), ('sobel_context_only', 'pa_context_only'),
    ('equal_context', 'pa_context_only'), ('equal_context', 'sobel_context_only'),
)
MATCHER = prior.ROUND7 / 'unified_bsds_evaluator.py'
MATCHER_BINARIES = tuple(sorted((prior.OLD_CACHE / 'tools/py-bsds500/bsds').glob('correspond_pixels*.so')))


def inputs():
    source_file = prior.OUT / 'source_frozen.json'
    runtime_file = prior.OUT / 'target_runtime.json'
    admission_file = prior.OUT / 'target_admission.json'
    prior_source_protocol = prior.OUT / 'protocol.json'
    source, runtime, admission = (prior.read(file) for file in (source_file, runtime_file, admission_file))
    if source['protocol_sha256'] != prior.sha(prior_source_protocol):
        raise ValueError('Source protocol no longer matches the frozen source configurations')
    if runtime['runner_sha256'] != prior.sha(ENGINEERING / 'run.py'):
        raise ValueError('The prior operator/target runtime changed')
    prior_binding = hashlib.sha256(''.join(prior.sha(file) for file in
                                      (source_file, runtime_file, admission_file)).encode()).hexdigest()
    if prior.read(prior.OUT / 'results.json')['frozen_source_sha256'] != prior.sha(source_file):
        raise ValueError('Prior report is not bound to these source configurations')
    entries = sorted(admission['datasets'][prior.CID]['entries'], key=lambda item: item['sample_id'])
    if len(entries) != 38 or len({entry['sample_id'] for entry in entries}) != 38:
        raise ValueError('Unexpected admitted CID image set')
    if len(MATCHER_BINARIES) != 1:
        raise ValueError('Expected one pinned compiled correspondence matcher')
    frozen_files = (source_file, runtime_file, admission_file, MATCHER,
                    MATCHER_BINARIES[0], prior.ROUND7 / 'ablation_operator.py',
                    MATCHER_BINARIES[0].with_name('thin.py'))
    protocol = dict(
        source_of_truth={str(file): prior.sha(file) for file in frozen_files},
        prior_binding_sha256=prior_binding,
        thresholds=THRESHOLDS.tolist(), models=list(source['models']), arms=list(ARMS),
        comparisons=[list(item) for item in CONTRASTS], max_dist=.0075, nms_low=.02,
        binary_thinning=True, matched_pixel_seed=4517, paired_draws=5000,
        sample_ids=[entry['sample_id'] for entry in entries],
        count_order=['matched_prediction', 'prediction', 'matched_truth', 'truth'],
        per_arm_fixed_thresholds={model: {arm: settings[arm]['threshold'] for arm in ARMS}
                                  for model, settings in source['models'].items()},
        runner_sha256=prior.sha(__file__),
    )
    protocol_sha = prior.frozen(OUT / 'protocol.json', protocol)
    return source['models'], entries, prior_binding, protocol_sha


def grid_index(value):
    idx = int(np.argmin(abs(THRESHOLDS - value)))
    if abs(THRESHOLDS[idx] - value) > 1e-12:
        raise ValueError(f'Threshold {value} is outside the 99-point grid')
    return idx


def checkpoint(model, sample, selection):
    return OUT / 'counts' / model / sample / (selection + '.json')


def checkpoint_counts(path, binding):
    row = prior.checked_record(path, binding)
    if row is None:
        return None
    counts = np.asarray(row['counts'], dtype=np.float64)
    if (counts.shape != (99, 4) or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts[:, 0] > counts[:, 1]) or np.any(counts[:, 2] > counts[:, 3])
            or np.any(counts[:, 3] != counts[0, 3])):
        raise ValueError(f'Invalid checkpoint counts: {path}')
    return counts


def job_records(models, entries, prior_binding, protocol_sha):
    jobs = []
    seen = set()
    for arm in ARMS:
        for entry in entries:
            sample = entry['sample_id']
            for model, settings in models.items():
                setting = settings[arm]
                selection = setting['selected_name']
                key = model, sample, selection
                if key in seen:
                    continue
                seen.add(key)
                path = checkpoint(*key)
                jobs.append((model, entry, setting, arm, settings[prior.BASE]['config'],
                             prior_binding, protocol_sha, path))
    return jobs


def sweep(job):
    model, entry, setting, arm, base, prior_binding, protocol_sha, path = job
    sample = entry['sample_id']
    started = time.monotonic()
    fixed = prior.checked_record(prior.CACHE / 'counts/target' / prior.CID / model / f'{sample}.json', prior_binding)
    if fixed is None or fixed['sample_id'] != sample or fixed['annotations'] != 1:
        raise ValueError(f'Missing prior fixed-threshold receipt for {model}/{sample}')
    map_file = prior.CACHE / 'maps' / prior.CID / model / f'{sample}.npz'
    map_receipt = prior.checked_record(map_file.with_suffix('.json'), prior_binding)
    if map_receipt is None or prior.sha(map_file) != map_receipt['map_sha256']:
        raise ValueError(f'Invalid frozen prediction map for {model}/{sample}')
    if prior.sha(entry['gt']) != entry['gt_sha256']:
        raise ValueError(f'Admitted CID label bytes changed for {sample}')
    with np.load(map_file, allow_pickle=False) as arrays:
        probability, sobel = arrays['prob'], arrays['edge']
    labels = prior.target_labels(prior.CID, entry['gt'])
    fields = prior.op.prepared(probability, sobel, base)
    score = prior.score(fields, setting['config'])
    counts = prior.match(score, labels, THRESHOLDS, .0075)
    if counts[grid_index(setting['threshold'])] != fixed['counts'][arm]:
        raise ValueError(f'99-grid fixed threshold slice differs for {model}/{sample}/{arm}')
    elapsed = time.monotonic() - started
    prior.atomic(path, dict(binding_sha256=protocol_sha, model=model, sample_id=sample,
                            selection=setting['selected_name'], map_sha256=map_receipt['map_sha256'],
                            fixed_threshold_parity=True, elapsed_seconds=elapsed, counts=counts))
    return model, sample, arm, elapsed


def status(state, phase, **values):
    from datetime import datetime, timezone
    record = dict(status=state, phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    prior.atomic(OUT / 'status.json', record)
    print(record, flush=True)


def report(models, entries, protocol_sha):
    from metrics import paired_ap_difference, summarize_counts
    import metrics

    reports = {}
    for model, settings in models.items():
        counts = {}
        for arm in ARMS:
            candidate = settings[arm]['selected_name']
            counts[arm] = np.asarray([checkpoint_counts(checkpoint(model, entry['sample_id'], candidate), protocol_sha)
                                      for entry in entries])
        arms = {arm: summarize_counts(counts[arm], THRESHOLDS, settings[arm]['threshold']) for arm in ARMS}
        pairs = []
        for first, second in CONTRASTS:
            pairs.append(dict(first=first, second=second,
                              **paired_ap_difference(counts[first], counts[second])))
        reports[model] = dict(n_images=len(entries), arms=arms, comparisons=pairs)
    prior.atomic(OUT / 'results.json', dict(
        protocol_sha256=protocol_sha, shared_metric_sha256=prior.sha(metrics.__file__), target=prior.CID,
        threshold_policy='Deployment F uses original per-arm source-fixed thresholds; ODS/OIS are target benchmark summaries only.',
        comparator_scope='Exact same admitted images, frozen maps, score configs, and matching as the existing CID fixed-F analysis.',
        paired_interval='Percentile 95%, 5000 paired image draws, unadjusted, conditional on source freeze.',
        reports=reports,
    ))


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models, entries, prior_binding, protocol_sha = inputs()
        jobs = job_records(models, entries, prior_binding, protocol_sha)
        if args.pilot:
            chosen = {entries[len(entries) // 3]['sample_id'], entries[2 * len(entries) // 3]['sample_id']}
            jobs = [job for job in jobs if job[1]['sample_id'] in chosen and job[3] in ARMS[:2]]
        pending = [job for job in jobs if checkpoint_counts(job[-1], protocol_sha) is None]
        reused = len(jobs) - len(pending)
        if not args.pilot:
            status('running', '99 阈值逐图匹配', completed=reused, total=len(jobs), eta_seconds=None,
                   reused=reused, workers=args.workers)
        started = time.monotonic()
        timings = []
        if pending:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                futures = [pool.submit(sweep, job) for job in pending]
                for future in as_completed(futures):
                    model, sample, arm, elapsed = future.result()
                    timings.append(dict(model=model, sample=sample, arm=arm, seconds=elapsed))
                    n = len(timings)
                    wall = time.monotonic() - started
                    if not args.pilot:
                        eta = round(wall / n * (len(pending) - n)) if n >= min(args.workers, len(pending)) else None
                        status('running', '99 阈值逐图匹配', completed=reused + n, total=len(jobs),
                               eta_seconds=eta, reused=reused, workers=args.workers, last_model=model,
                               last_image=sample, last_arm=arm)
                    else:
                        print(f'{model}/{sample}/{arm}: {elapsed:.2f} seconds', flush=True)
        if args.pilot:
            prior.atomic(OUT / 'pilot.json', dict(n_completed=len(jobs), reused=reused,
                         timings=timings, wall_seconds=time.monotonic() - started, workers=args.workers))
            return
        status('running', '汇总 AP、ODS/OIS 和配对区间', completed=len(jobs), total=len(jobs), eta_seconds=None)
        report(models, entries, protocol_sha)
        status('complete', 'AP、ODS/OIS 和配对区间已落盘', completed=len(jobs), total=len(jobs), eta_seconds=0,
               report=str(OUT / 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        main(args)
    except Exception as error:
        if not args.pilot and not isinstance(error, BlockingIOError):
            status('failed', '计算中断', error=repr(error), traceback=traceback.format_exc(), eta_seconds=None)
        raise
