"""Evaluate source-selected nonzero HED context amplitudes on the existing CID split."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import importlib.util
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
OLD_RUNNER = ROOT / 'work/hed_rbcm_20261110/target/run.py'
OUT = ROOT / 'outputs/RBCM_HED_NONZERO_20261110/target'
SOURCE = OUT.parent / 'source'

spec = importlib.util.spec_from_file_location('rbc_previous_hed_target', OLD_RUNNER)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
hed = prior.hed

import numpy as np

RING = 'equal_context'
PLAIN = 'plain_identity'
BASE = 'calibrated_base'
BOX = 'smoothing_equal_context'
LEGACY = 'legacy_main'
PA = 'pa_context_only'
SOBEL = 'sobel_context_only'
GAUSSIAN = 'gaussian_equal_context'
ACTIVE = (RING, BOX, PA, SOBEL, GAUSSIAN)
PRIMARY = tuple((RING, other) for other in (PLAIN, BASE, BOX, PA, SOBEL, GAUSSIAN))
SECONDARY = ((BOX, PLAIN), (BOX, BASE), (PA, BASE), (SOBEL, BASE), (GAUSSIAN, BASE),
             (SOBEL, PA), (LEGACY, PLAIN), (LEGACY, BOX), (RING, LEGACY))


def inputs():
    entries, original_tau, bare_binding = prior.read_target_inputs()
    old_source, old_freeze = prior.read_source_freeze()
    old_protocol_path = prior.OUT / 'protocol.json'
    old_protocol = hed.read(old_protocol_path)
    old_binding = hed.digest(old_protocol_path)
    old_result = hed.read(prior.OUT / 'results.json')
    if (old_result['target_protocol_sha256'] != old_binding or old_protocol['source_frozen_sha256'] != old_freeze
            or old_protocol['previous_bare_source_sha256'] != bare_binding
            or old_protocol['code_sha256'] != hed.digest(OLD_RUNNER)
            or old_result['sample_ids'] != [entry['sample_id'] for entry in entries]
            or old_protocol['thresholds'] != prior.THRESHOLDS.tolist()):
        raise ValueError('Previous target curves and HED map protocol do not bind to the admitted CID split')
    for name, digest in old_protocol['dependencies'].items():
        if hed.digest(name) != digest:
            raise ValueError(f'Frozen evaluator or score dependency changed: {name}')
    source_path = SOURCE / 'source_frozen.json'
    source_protocol_path = SOURCE / 'protocol.json'
    source_protocol = hed.read(source_protocol_path)
    frozen = hed.read(source_path)
    if (frozen.get('target_score_accessed_for_selection') is not False
            or frozen['protocol_sha256'] != hed.digest(source_protocol_path)
            or frozen['prior_source_frozen_sha256'] != old_freeze
            or source_protocol['prior_source_protocol_sha256'] != hed.digest(prior.SOURCE / 'protocol.json')
            or source_protocol['hed_provenance']['weights']['sha256'] != hed.WEIGHTS_SHA):
        raise ValueError('Nonzero receipt does not match the frozen source/official HED')
    selected = frozen['selected']
    if set(selected) != set(prior.ARMS) or any(selected[arm] != old_source[arm] for arm in (PLAIN, BASE, LEGACY)):
        raise ValueError('Nonzero sensitivity must preserve bare HED, calibration, and legacy settings')
    if any(selected[arm]['config']['multiplier'] not in (.5, 1.) for arm in ACTIVE):
        raise ValueError('Every active comparison must use one of the predeclared nonzero amplitudes')
    if selected[PLAIN]['threshold'] != original_tau:
        raise ValueError('Bare HED source threshold changed')
    for setting in selected.values():
        prior.threshold_index(setting['threshold'])
    # Reuse a complete curve only when the same exact scoring config was evaluated.
    reused = {}
    for arm in prior.ARMS:
        matches = {item['selected_name'] for item in old_source.values() if selected[arm]['config'] == item['config']}
        if len(matches) == 1:
            reused[arm] = matches.pop()
        elif matches:
            raise ValueError(f'Ambiguous previous curve for {arm}')
    freeze_hash = hed.digest(source_path)
    binding = hed.freeze(OUT / 'protocol.json', dict(
        runner_sha256=hed.digest(__file__), previous_target_runner_sha256=hed.digest(OLD_RUNNER),
        previous_target_protocol_sha256=old_binding, source_frozen_sha256=freeze_hash,
        previous_bare_source_sha256=bare_binding, admitted_ids=old_result['sample_ids'],
        thresholds=prior.THRESHOLDS.tolist(), score_map_reuse=True, reused_arms=reused,
        paired_resamples=prior.NBOOT, paired_seed=prior.SEED, no_target_selection=True,
        primary_contrasts=[list(pair) for pair in PRIMARY],
    ))
    return entries, selected, reused, old_binding, bare_binding, binding, freeze_hash


def path_for(sample, arm):
    return OUT / 'counts' / sample / f'{arm}.json'


def load_fresh(sample, arm, binding):
    path = path_for(sample, arm)
    if not path.exists():
        return None
    row = hed.read(path)
    if row['binding_sha256'] != binding or row['sample_id'] != sample or row['arm'] != arm:
        raise ValueError(f'Stale sensitivity checkpoint: {path}')
    return prior.checked_counts(row['counts'], path)


def read_map(sample, old_binding):
    path = prior.OUT / 'maps' / f'{sample}.npz'
    receipt = hed.read(path.with_suffix('.json'))
    if (receipt['binding_sha256'] != old_binding or receipt['sample_id'] != sample
            or receipt.get('original_HED_fixed_threshold_count_parity') is not True
            or receipt['map_sha256'] != hed.digest(path)):
        raise ValueError(f'Previous raw probability/Sobel map failed integrity check for {sample}')
    with np.load(path, allow_pickle=False) as saved:
        return saved['probability'], saved['sobel']


def sweep_image(entry, jobs, base, old_binding, binding):
    sample = entry['sample_id']
    started = time.monotonic()
    probability, sobel = read_map(sample, old_binding)
    labels = prior.load_target(entry)
    fields = prior.prepared(probability, sobel, base)
    for arm, config in jobs:
        path = path_for(sample, arm)
        counts = hed.match(prior.score(fields, config), labels, prior.THRESHOLDS, .0075)
        prior.checked_counts(counts, path)
        hed.save(path, dict(binding_sha256=binding, sample_id=sample, arm=arm, counts=counts))
    return sample, len(jobs), time.monotonic() - started


def assemble_counts(entries, selected, reused, old_binding, bare_binding, binding):
    result = {}
    for arm in selected:
        if arm == PLAIN:
            rows = [prior.read_bare(entry, bare_binding) for entry in entries]
        elif arm in reused:
            rows = [prior.load_checkpoint(entry['sample_id'], reused[arm], old_binding) for entry in entries]
        else:
            rows = [load_fresh(entry['sample_id'], arm, binding) for entry in entries]
        if any(row is None for row in rows):
            raise ValueError(f'Missing image curve for {arm}')
        result[arm] = np.asarray(rows)
    return result


def report(entries, selected, reused, old_binding, bare_binding, binding, freeze_hash):
    counts = assemble_counts(entries, selected, reused, old_binding, bare_binding, binding)
    n = len(entries)
    weights = np.random.default_rng(prior.SEED).multinomial(n, np.full(n, 1 / n), size=prior.NBOOT)
    point = {arm: prior.summarize_counts(counts[arm], prior.THRESHOLDS, selected[arm]['threshold']) for arm in prior.ARMS}
    draws = {arm: prior.bootstrap_metrics(counts[arm], selected[arm]['threshold'], weights, False) for arm in prior.ARMS}
    pairs = []
    for first, second in (*PRIMARY, *SECONDARY):
        metrics = {}
        for key in ('AP_BSDS_101point', 'F_fixed'):
            low, high = np.quantile(draws[first][key] - draws[second][key], [.025, .975])
            metrics[key] = dict(delta=point[first][key] - point[second][key],
                                ci95_paired_image_low=float(low), ci95_paired_image_high=float(high))
        pairs.append(dict(first=first, second=second, primary=(first, second) in PRIMARY, metrics=metrics))
    result = dict(dataset=hed.CID, n_images=n, sample_ids=[entry['sample_id'] for entry in entries],
                  arms=point, comparisons=pairs, source_selected=selected, reused_arms=reused,
                  source_frozen_sha256=freeze_hash, target_protocol_sha256=binding,
                  design='Source-only nonzero sensitivity: m=.5 or 1 for matched context families.',
                  interval='95% percentile, 5000 paired whole-image draws, unadjusted.')
    hed.save(OUT / 'results.json', result)
    lines = [f'HED 非零幅度敏感性 · Groningen CID {n} 张',
             '所有上下文比较在 BIPED 源开发集从 m=.5 或 1 中锁定幅度与阈值。',
             'AP、ODS、OIS、部署 F 为百分数。ODS/OIS 使用 CID 曲线，部署 F 使用源阈值。',
             '', '方法\t源修正幅度\t源部署阈值\tAP\tODS\tOIS\t部署 F']
    for arm, metrics in point.items():
        setting = selected[arm]
        fields = [prior.LABEL[arm], f'{setting["config"]["multiplier"]:g}', f'{setting["threshold"]:.2f}']
        fields += [f'{100 * metrics[key]:.3f}' for key in prior.METRIC_LABEL]
        lines.append('\t'.join(fields))
    lines += ['', '差值为前者减后者，单位为百分点。95% 区间为 5,000 次逐图配对重采样。',
              '类别\t比较\t指标\t差值\t95% 下界\t95% 上界']
    for pair in pairs:
        label = f'{prior.LABEL[pair["first"]]} − {prior.LABEL[pair["second"]]}'
        for metric, values in pair['metrics'].items():
            fields = ['主要' if pair['primary'] else '次要', label, prior.METRIC_LABEL[metric]]
            fields += [f'{100 * values[key]:+.3f}' for key in ('delta', 'ci95_paired_image_low', 'ci95_paired_image_high')]
            lines.append('\t'.join(fields))
    hed.atomic(OUT / 'CID非零幅度汇总.txt', ('\n'.join(lines) + '\n').encode('utf-8'))


def state(status='running', **fields):
    record = dict(status=status, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    hed.save(OUT / 'status.json', record)
    print(record, flush=True)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries, selected, reused, old_binding, bare_binding, binding, freeze_hash = inputs()
        if args.pilot:
            entry = entries[len(entries) // 2]
            jobs = [(RING, selected[RING]['config'])] if load_fresh(entry['sample_id'], RING, binding) is None else []
            sample, done, seconds = sweep_image(entry, jobs, selected[BASE]['config'], old_binding, binding)
            probe = dict(sample_id=sample, arm=RING, fresh_sweeps=done, seconds=seconds, binding_sha256=binding)
            hed.save(OUT / 'pilot.json', probe)
            print(probe, flush=True)
            return
        pending = []
        for entry in entries:
            sample = entry['sample_id']
            jobs = [(arm, selected[arm]['config']) for arm in prior.ARMS
                    if arm not in reused and load_fresh(sample, arm, binding) is None]
            if jobs:
                pending.append((entry, jobs))
        total = len(entries) * len(selected)
        previous = len(entries) * len(reused)
        todo = sum(len(jobs) for _, jobs in pending)
        completed = total - todo
        state(phase='CID 99 阈值非零幅度匹配', completed=completed, total=total, reused_curves=previous,
              fresh_sweeps=0, workers=args.workers, eta_seconds=None)
        if pending:
            started = time.monotonic()
            fresh = 0
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                tasks = [pool.submit(sweep_image, entry, jobs, selected[BASE]['config'], old_binding, binding)
                         for entry, jobs in pending]
                for index, task in enumerate(as_completed(tasks), start=1):
                    sample, done, seconds = task.result()
                    fresh += done
                    eta = round((time.monotonic() - started) * (todo - fresh) / fresh) if index >= min(4, len(pending)) else None
                    state(phase='CID 99 阈值非零幅度匹配', completed=completed + fresh, total=total,
                          reused_curves=previous, fresh_sweeps=fresh, workers=args.workers, eta_seconds=eta,
                          last_image=sample, last_image_seconds=seconds)
        state(phase='汇总 AP、部署 F 与配对区间', completed=total, total=total, eta_seconds=None)
        report(entries, selected, reused, old_binding, bare_binding, binding, freeze_hash)
        state('complete', phase='非零幅度对照已落盘', completed=total, total=total, eta_seconds=0,
              report=str(OUT / 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--pilot', action='store_true', help='Evaluate one source-locked ring score and save its curve')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        main(args)
    except Exception as error:
        if not args.pilot and not isinstance(error, BlockingIOError):
            state('failed', phase='非零幅度匹配中断', error=repr(error), traceback=traceback.format_exc(), eta_seconds=None)
        raise
