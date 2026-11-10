"""Apply BIPED-frozen RBCM arms to the author HED checkpoint on CID."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import importlib.util
from io import BytesIO
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / 'work/cid_ap_hed_20261109'
ADAPTER = PREVIOUS / 'hed/run.py'
SOURCE = ROOT / 'outputs/RBCM_HED_AUG_20261110/source'
OUT = SOURCE.parent / 'target'
MATCHER_DIR = Path('/mnt/h/2026/RBCM_round7_20261031/tools/py-bsds500/bsds')

spec = importlib.util.spec_from_file_location('rbc_hed_author_adapter', ADAPTER)
hed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hed)

import numpy as np
from ablation_operator import prepared
from backbone_runtime import read_rgb, sobel_from_rgb
from metrics import THRESHOLDS, summarize_counts
from run import score
from unified_bsds_evaluator import summarize_counts as summarize_round7

ARMS = ('plain_identity', 'calibrated_base', 'legacy_main', 'equal_context',
        'smoothing_equal_context', 'pa_context_only', 'sobel_context_only', 'gaussian_equal_context')
PRIMARY = (('equal_context', 'plain_identity'), ('equal_context', 'smoothing_equal_context'))
SECONDARY = (
    ('equal_context', 'calibrated_base'), ('calibrated_base', 'plain_identity'),
    ('legacy_main', 'plain_identity'), ('legacy_main', 'smoothing_equal_context'),
    ('smoothing_equal_context', 'plain_identity'), ('pa_context_only', 'plain_identity'),
    ('sobel_context_only', 'plain_identity'), ('sobel_context_only', 'pa_context_only'),
    ('equal_context', 'pa_context_only'), ('equal_context', 'sobel_context_only'),
    ('equal_context', 'gaussian_equal_context'),
)
SEED = 4517
NBOOT = 5000
LABEL = {
    'plain_identity': '作者版 HED', 'calibrated_base': '仅校准', 'legacy_main': '旧 RBCM',
    'equal_context': '等权环形融合', 'smoothing_equal_context': '等预算盒平滑',
    'pa_context_only': '仅 Pa 上下文', 'sobel_context_only': '仅 Sobel 上下文',
    'gaussian_equal_context': '尺度匹配高斯',
}
METRIC_LABEL = {'AP_BSDS_101point': 'AP', 'ODS': 'ODS', 'OIS': 'OIS', 'F_fixed': '部署 F'}


def read_target_inputs():
    """Verify the exact adapter, checkpoint, labels and existing bare-HED protocol."""
    prior_protocol = hed.read(hed.OUT / 'protocol.json')
    old_source = hed.read(hed.OUT / 'source_frozen.json')
    provenance = prior_protocol['provenance']
    for path, expected in (
        (ADAPTER, prior_protocol['runner_sha256']),
        (hed.ORIGINAL_DEPLOY, provenance['original_deploy_sha256']),
        (hed.COMPAT_DEPLOY, provenance['compatible_deploy_sha256']),
        (hed.WEIGHTS, provenance['weights']['sha256']),
        (hed.ADMISSION, prior_protocol['target_admission_sha256']),
        (hed.OUT / 'protocol.json', old_source['protocol_sha256']),
        (hed.PREVIOUS / 'run.py', prior_protocol['matcher']['previous_runner_sha256']),
        (ROOT / 'work/round7_20261031/unified_bsds_evaluator.py', prior_protocol['matcher']['unified_matcher_sha256']),
        (PREVIOUS / 'metrics.py', prior_protocol['metrics_sha256']),
    ):
        if hed.digest(path) != expected:
            raise ValueError(f'Previous HED input/implementation changed: {path}')
    if provenance['opencv_version'] != hed.cv2.__version__:
        raise ValueError('OpenCV runtime differs from the previous HED evaluation')
    entries = sorted(hed.read(hed.ADMISSION)['datasets'][hed.CID]['entries'], key=lambda entry: entry['sample_id'])
    bare_source_sha = hed.digest(hed.OUT / 'source_frozen.json')
    earlier_report = hed.read(hed.OUT / 'results.json')
    if (len(entries) != 38 or len({row['sample_id'] for row in entries}) != 38
            or [row['sample_id'] for row in entries] != earlier_report['target_sample_ids']
            or bare_source_sha != earlier_report['frozen_source_sha256']):
        raise ValueError('Previously admitted CID sample IDs or original HED threshold differ')
    return entries, old_source['threshold'], bare_source_sha


def read_source_freeze():
    path = SOURCE / 'source_frozen.json'
    frozen = hed.read(path)
    protocol = hed.read(SOURCE / 'protocol.json')
    if (frozen.get('target_score_accessed_for_selection') is not False
            or protocol.get('target_score_accessed_for_selection') is not False
            or frozen['protocol_sha256'] != hed.digest(SOURCE / 'protocol.json')
            or protocol['prior_hed_frozen_sha256'] != hed.digest(hed.OUT / 'source_frozen.json')
            or protocol['hed_provenance']['weights']['sha256'] != hed.WEIGHTS_SHA):
        raise ValueError('Missing or inconsistent BIPED-only freeze receipt')
    selected = frozen['selected']
    if set(selected) != set(ARMS):
        raise ValueError(f'Source-selected arm list changed: {list(selected)}')
    for setting in selected.values():
        threshold_index(setting['threshold'])
    return selected, hed.digest(path)


def threshold_index(threshold):
    position = int(np.argmin(abs(THRESHOLDS - threshold)))
    if abs(THRESHOLDS[position] - threshold) > 1e-12:
        raise ValueError(f'Source threshold {threshold} is outside the target grid')
    return position


def distinct_transforms(selected):
    unique = {}
    for arm in ARMS:
        config = selected[arm]['config']
        identity = selected[arm]['selected_name']
        if arm == 'plain_identity':
            if identity != 'plain_identity' or config['family'] != 'plain_identity' or config['multiplier'] != 0:
                raise ValueError('The reused control must be bare HED')
            continue
        if identity in unique and unique[identity] != config:
            raise ValueError(f'Source receipt assigned differing configs the same name: {identity}')
        unique[identity] = config
    return unique


def checked_counts(counts, path):
    value = np.asarray(counts, dtype=np.float64)
    if (value.shape != (99, 4) or not np.isfinite(value).all() or np.any(value < 0)
            or np.any(value[:, ::2] > value[:, 1::2]) or np.any(value[:, 3] != value[0, 3])):
        raise ValueError(f'Invalid 99-point counts in {path}')
    return value


def read_bare(entry, bare_binding):
    sample = entry['sample_id']
    path = hed.CACHE / 'counts/target_cid' / f'{sample}.json'
    row = hed.read(path)
    if (row['sample_id'] != sample or row['binding_sha256'] != bare_binding
            or row['image_sha256'] != entry['image_sha256'] or row['gt_sha256'] != entry['gt_sha256']
            or row['thresholds'] != THRESHOLDS.tolist()):
        raise ValueError(f'Bare-HED counts do not match the admitted CID entry: {path}')
    return checked_counts(row['counts'], path)


def load_target(entry):
    if hed.digest(entry['image']) != entry['image_sha256'] or hed.digest(entry['gt']) != entry['gt_sha256']:
        raise ValueError(f'CID bytes changed for {entry["sample_id"]}')
    return hed.target_labels(hed.CID, entry['gt'])


def infer_and_verify(entry, labels, baseline, tau, full=False):
    probability = hed.infer(entry['image'])
    index = threshold_index(tau)
    thresholds = THRESHOLDS if full else THRESHOLDS[[index]]
    expected = baseline if full else baseline[[index]]
    if not np.array_equal(hed.match(probability, labels, thresholds, .0075), expected):
        raise ValueError(f'HED inference failed parity with previous CID counts for {entry["sample_id"]}')
    return probability


def read_or_cache_input(entry, labels, bare, tau, binding):
    sample = entry['sample_id']
    path = OUT / 'maps' / f'{sample}.npz'
    meta = path.with_suffix('.json')
    if meta.exists():
        receipt = hed.read(meta)
        if receipt['binding_sha256'] != binding or hed.digest(path) != receipt['map_sha256']:
            raise ValueError(f'Invalid target map receipt for {sample}')
        with np.load(path, allow_pickle=False) as saved:
            return saved['probability'], saved['sobel']
    probability = infer_and_verify(entry, labels, bare, tau)
    sobel = sobel_from_rgb(read_rgb(entry['image']))
    with BytesIO() as buffer:
        np.savez(buffer, probability=probability, sobel=sobel)
        hed.atomic(path, buffer.getvalue())
    hed.save(meta, dict(binding_sha256=binding, map_sha256=hed.digest(path), sample_id=sample,
                        original_HED_fixed_threshold_count_parity=True))
    return probability, sobel


def count_path(sample, candidate):
    return OUT / 'counts' / sample / f'{candidate}.json'


def load_checkpoint(sample, candidate, binding):
    path = count_path(sample, candidate)
    if not path.exists():
        return None
    row = hed.read(path)
    if (row['sample_id'] != sample or row['candidate'] != candidate or row['binding_sha256'] != binding):
        raise ValueError(f'Invalid target checkpoint identity/protocol: {path}')
    return checked_counts(row['counts'], path)


def evaluate_image(entry, pending, base, tau, bare_binding, binding):
    started = time.monotonic()
    sample = entry['sample_id']
    labels = load_target(entry)
    bare = read_bare(entry, bare_binding)
    probability, sobel = read_or_cache_input(entry, labels, bare, tau, binding)
    fields = prepared(probability, sobel, base)
    for candidate, config in pending:
        path = count_path(sample, candidate)
        counts = hed.match(score(fields, config), labels, THRESHOLDS, .0075)
        checked_counts(counts, path)
        hed.save(path, dict(sample_id=sample, candidate=candidate, binding_sha256=binding, counts=counts))
    return sample, len(pending), time.monotonic() - started


def prf(counts):
    mp_, p, mt, t = np.moveaxis(counts, -1, 0)
    precision = mp_ / np.maximum(p, 1)
    recall = mt / np.maximum(t, 1)
    return precision, recall, f_score(precision, recall)


def f_score(precision, recall):
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def bootstrap_metrics(counts, threshold, weights, oracles):
    shape = (len(weights), len(THRESHOLDS), 4)
    curves = (weights @ counts.reshape(len(counts), -1)).reshape(shape)
    _, _, fixed = prf(curves[:, threshold_index(threshold)])
    ap = np.fromiter((summarize_round7(curve, THRESHOLDS)['AP_BSDS_101point'] for curve in curves),
                     dtype=float, count=len(curves))
    result = dict(AP_BSDS_101point=ap, F_fixed=fixed)
    if oracles:
        _, _, image_f = prf(counts)
        picked = counts[np.arange(len(counts)), image_f.argmax(axis=1)]
        _, _, result['OIS'] = prf(weights @ picked)
        fraction = np.linspace(0., 1., 100)
        ods = np.empty(len(weights))
        for index in range(0, len(curves), 64):
            batch = curves[index:index + 64]
            p, r, _ = prf(batch)
            p = p[:, :-1, None] * (1 - fraction) + p[:, 1:, None] * fraction
            r = r[:, :-1, None] * (1 - fraction) + r[:, 1:, None] * fraction
            ods[index:index + len(batch)] = f_score(p, r).max(axis=(1, 2))
        result['ODS'] = ods
    return result


def text_report(result):
    lines = [
        f'作者版 HED 接入 RBCM · Groningen CID {result["n_images"]} 张',
        'AP、ODS、OIS 与部署 F 为百分数。部署阈值和修正幅度由 BIPED 源开发集锁定。',
        'ODS/OIS 使用 CID 的阈值曲线。作者版 HED 的原 99 点计数直接复用。',
        '', '方法\t源修正幅度\t源部署阈值\tAP\tODS\tOIS\t部署 F',
    ]
    for arm, metrics in result['arms'].items():
        setting = result['source_selected'][arm]
        row = [LABEL[arm], f'{setting["config"]["multiplier"]:g}', f'{setting["threshold"]:.2f}']
        row += [f'{100 * metrics[key]:.3f}' for key in METRIC_LABEL]
        lines.append('\t'.join(row))
    lines += [
        '', '差值为前者减后者，单位为百分点。95% 逐图配对区间，5,000 次重采样，未校正多重比较。',
        '每个重采样中重新计算 ODS，OIS 按每张图的离散最佳点汇总计数。',
        '类别\t比较\t指标\t差值\t95% 下界\t95% 上界',
    ]
    for pair in result['comparisons']:
        comparison = f'{LABEL[pair["first"]]} − {LABEL[pair["second"]]}'
        for key, values in pair['metrics'].items():
            row = ['主要' if pair['primary'] else '次要', comparison, METRIC_LABEL[key]]
            row += [f'{100 * values[name]:+.3f}' for name in
                    ('delta', 'ci95_paired_image_low', 'ci95_paired_image_high')]
            lines.append('\t'.join(row))
    hed.atomic(OUT / 'CID结果汇总.txt', ('\n'.join(lines) + '\n').encode('utf-8'))


def summarize(entries, selected, bare_binding, binding, freeze_sha):
    counts = {'plain_identity': np.asarray([read_bare(entry, bare_binding) for entry in entries])}
    by_transform = {}
    for arm in ARMS[1:]:
        key = selected[arm]['selected_name']
        if key not in by_transform:
            rows = [load_checkpoint(entry['sample_id'], key, binding) for entry in entries]
            if any(row is None for row in rows):
                raise ValueError(f'Missing target checkpoint for {arm}')
            by_transform[key] = np.asarray(rows)
        counts[arm] = by_transform[key]
    metrics = {arm: summarize_counts(counts[arm], THRESHOLDS, selected[arm]['threshold']) for arm in ARMS}
    previous = hed.read(hed.OUT / 'results.json')['metrics']
    for key in ('F_fixed', 'ODS', 'OIS', 'AP_BSDS_101point'):
        if abs(metrics['plain_identity'][key] - previous[key]) > 1e-12:
            raise ValueError(f'Reused bare-HED summary failed parity for {key}')
    n = len(entries)
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=NBOOT)
    oracle_arms = {arm for pair in PRIMARY for arm in pair}
    draws = {arm: bootstrap_metrics(counts[arm], selected[arm]['threshold'], weights, arm in oracle_arms) for arm in ARMS}
    comparisons = []
    for first, second in (*PRIMARY, *SECONDARY):
        primary = (first, second) in PRIMARY
        keys = ('AP_BSDS_101point', 'F_fixed', 'ODS', 'OIS') if primary else ('AP_BSDS_101point', 'F_fixed')
        deltas = {}
        for key in keys:
            difference = draws[first][key] - draws[second][key]
            low, high = np.quantile(difference, [.025, .975])
            deltas[key] = dict(delta=metrics[first][key] - metrics[second][key],
                               ci95_paired_image_low=float(low), ci95_paired_image_high=float(high))
        comparisons.append(dict(first=first, second=second, primary=primary, metrics=deltas))
    result = dict(
        backbone='hed_author_bsds', dataset=hed.CID, n_images=n,
        sample_ids=[entry['sample_id'] for entry in entries], arms=metrics, comparisons=comparisons,
        source_frozen_sha256=freeze_sha, target_protocol_sha256=binding, source_selected=selected,
        existing_HED_bare_counts_reused=True,
        policy='Deployment F uses BIPED source lock; ODS/OIS are target benchmark summaries.',
        interval=dict(resamples=NBOOT, seed=SEED, unit='paired image', multiplicity_adjusted=False,
                      ODS='remaximize pooled interpolated F per bootstrap draw',
                      OIS='pool per-image discrete optimum counts with paired draw weights'),
    )
    hed.save(OUT / 'results.json', result)
    text_report(result)


def state(status='running', **fields):
    record = dict(status=status, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    hed.save(OUT / 'status.json', record)
    print(record, flush=True)


def pilot(entries, tau, bare_binding):
    entry = entries[len(entries) // 2]
    started = time.monotonic()
    labels = load_target(entry)
    original = read_bare(entry, bare_binding)
    raw = infer_and_verify(entry, labels, original, tau, full=True)
    probe = dict(sample_id=entry['sample_id'], shape=list(raw.shape), complete_99_point_bare_HED_parity=True,
                 old_bare_source_sha256=bare_binding, seconds=time.monotonic() - started)
    hed.save(OUT / 'adapter_pilot.json', probe)
    print(probe, flush=True)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    entries, tau, bare_binding = read_target_inputs()
    if args.pilot:
        pilot(entries, tau, bare_binding)
        return
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected, freeze_sha = read_source_freeze()
        if selected['plain_identity']['threshold'] != tau:
            raise ValueError('The source-locked bare-HED threshold must match the previous bare-HED threshold')
        transforms = distinct_transforms(selected)
        for entry in entries:
            read_bare(entry, bare_binding)
        dependencies = (ADAPTER, PREVIOUS / 'metrics.py', hed.PREVIOUS / 'run.py',
                        ROOT / 'work/round7_20261031/ablation_operator.py',
                        ROOT / 'work/round7_20261031/backbone/backbone_runtime.py',
                        ROOT / 'work/round7_20261031/unified_bsds_evaluator.py', MATCHER_DIR / 'thin.py')
        matcher_binary = tuple(MATCHER_DIR.glob('correspond_pixels*.so'))
        if len(matcher_binary) != 1:
            raise ValueError('Expected exactly one compiled BSDS matcher')
        binding = hed.freeze(OUT / 'protocol.json', dict(
            previous_bare_source_sha256=bare_binding, source_frozen_sha256=freeze_sha,
            admission_sha256=hed.digest(hed.ADMISSION), code_sha256=hed.digest(__file__),
            dependencies={str(path): hed.digest(path) for path in (*dependencies, *matcher_binary)},
            thresholds=THRESHOLDS.tolist(), max_dist=.0075, nms_low=.02, binary_thinning=True,
            paired_resamples=NBOOT, paired_seed=SEED, original_HED_checkpoint_sha256=hed.WEIGHTS_SHA,
            admitted_ids=[entry['sample_id'] for entry in entries], arms=list(ARMS),
        ))
        jobs = []
        total = len(entries) * (1 + len(transforms))
        for entry in entries:
            sample = entry['sample_id']
            pending = [(name, config) for name, config in transforms.items()
                       if load_checkpoint(sample, name, binding) is None]
            if pending:
                jobs.append((entry, pending))
        reused = total - sum(len(pending) for _, pending in jobs)
        state(phase='CID 99 阈值匹配', completed=reused, total=total, eta_seconds=None, workers=args.workers,
              old_HED_curves_reused=len(entries), selected_unique_corrections=len(transforms))
        if jobs:
            started = time.monotonic()
            finished = fresh = 0
            base = selected['calibrated_base']['config']
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                tasks = [pool.submit(evaluate_image, entry, pending, base, tau, bare_binding, binding)
                         for entry, pending in jobs]
                for task in as_completed(tasks):
                    sample, sweeps, seconds = task.result()
                    finished += 1
                    fresh += sweeps
                    done = reused + fresh
                    eta = round((time.monotonic() - started) * (total - done) / fresh) if finished >= min(4, len(jobs)) else None
                    state(phase='CID 99 阈值匹配', completed=done, total=total, eta_seconds=eta,
                          last_image=sample, last_image_seconds=seconds, workers=args.workers,
                          old_HED_curves_reused=len(entries), fresh_sweeps=fresh)
        state(phase='汇总 AP、ODS/OIS、固定 F 与配对区间', completed=total, total=total, eta_seconds=None)
        summarize(entries, selected, bare_binding, binding, freeze_sha)
        state('complete', phase='已汇总 HED+RBCM 对照', completed=total, total=total, eta_seconds=0,
              report=str(OUT / 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--pilot', action='store_true', help='Check one existing bare-HED 99-point CID receipt only')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        main(args)
    except Exception as error:
        if not args.pilot and not isinstance(error, BlockingIOError):
            state('failed', phase='目标集计算中断', error=repr(error), traceback=traceback.format_exc(), eta_seconds=None)
        raise
