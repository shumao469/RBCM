"""Fit matched correction arms on historical source development, then transfer them frozen."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
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
ROUND7 = ROOT / 'work/round7_20261031'
RELEASE = ROOT / 'work/github_release_snapshot'
OLD_OUT = ROOT / 'outputs/RBCM_round7_20261031'
OLD_CACHE = Path('/mnt/h/2026/RBCM_round7_20261031')
OLD_STAGE = Path('/mnt/h/2026/RBCM_review_20261028')
OUT = ROOT / 'outputs/RBCM_validation_20261108/engineering'
CACHE = Path('/mnt/h/2026/RBCM_validation_20261108/engineering')
sys.path[:0] = [str(ROUND7), str(ROUND7 / 'backbone'), str(RELEASE), str(RELEASE / 'src')]

import ablation_operator as op
from admission import adjudicate, berkeley, groningen, prior_images, sha
import cv2
import numpy as np

MODELS = {'edgeanchor_nyudv2': 'nyudv2', 'teed_biped': 'biped'}
THRESHOLD_INDEX = (38, 48, 58, 68, 78, 88)
THRESHOLDS = np.linspace(.01, .99, 99)[np.asarray(THRESHOLD_INDEX) - 1]
EXISTING = ('pa_context_only', 'sobel_context_only', 'equal_context',
            'smoothing_equal_context', 'pointwise_sobel_global_reference')
NEW = {'gaussian_equal_context': 'matched_gaussian', 'shifted_sobel_equal_context': 'translated_sobel_ring'}
AMPLITUDES = (.5, 1.)
BASE = 'calibrated_base'
CID = 'groningen_cid'
BSDS = 'bsds_official_train200'
COMPARE = (
    ('sobel_context_only', BASE), ('sobel_context_only', 'pa_context_only'),
    ('equal_context', 'pa_context_only'), ('equal_context', 'sobel_context_only'),
    ('sobel_context_only', 'pointwise_sobel_global_reference'),
    ('equal_context', 'smoothing_equal_context'), ('equal_context', 'gaussian_equal_context'),
    ('equal_context', 'shifted_sobel_equal_context'), ('equal_context', 'gaussian_at_ring_setting'),
    ('equal_context', 'shifted_at_ring_setting'),
)
_BACKBONES = {}
_MATCHERS = None
ORIGINAL_SOURCE_RUNNER_SHA = '91cbcfc412d0b08e2add6ace3a13f735cb10aac4462e6e90da683884fd21427c'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Resume mismatch at {path}')
    else:
        atomic(path, value)
    return sha(path)


def state(status='running', **fields):
    progress = dict(status=status, updated_at=datetime.now(timezone.utc).isoformat(),
                    updated_epoch=time.time(), pid=os.getpid(), **fields)
    atomic(OUT / 'status.json', progress)
    print(json.dumps(progress, ensure_ascii=False), flush=True)


def candidate_name(family, amplitude):
    return family + '_m' + str(amplitude).replace('.', 'p')


def source_entries(model):
    source = MODELS[model]
    if source == 'biped':
        path = OLD_OUT / 'backbone/teed_biped_source_biped_cache_manifest.json'
        document = read(path)
        selected = [row for row in document['entries'] if row['partition'] == 'development']
        out = []
        for entry in selected:
            sid = entry['sample_id']
            out.append(dict(entry, prior_counts=str(OLD_CACHE / f'evaluation_counts/source_development/teed_biped/{sid}.json')))
        return sorted(out, key=lambda entry: entry['sample_id'])
    manifest = read(OLD_STAGE / 'source_cache/nyudv2/manifest.json')
    entries = {row['sample_id']: row for row in manifest['entries']}
    split = read(ROOT / 'outputs/RBCM_round3_20261029/engineering/source_splits/nyudv2.json')
    development = [row for row in split['rows'] if row['partition'] == 'development']
    # The first twelve in the already-frozen salted group hash order are a common source budget for all arms.
    return [dict(entries[row['sample_id']], prior_counts=str(OLD_CACHE / f'ablation/one_to_one_counts/development/nyudv2/{index:06d}.json'))
            for index, row in enumerate(development[:12])]


def configurations(model):
    source = MODELS[model]
    old = read(ROOT / 'outputs/RBCM_round3_20261029/engineering/frozen_release_parameters.json')
    configs = op.build_candidate_configs(old[source]['candidates']['main_surround'], source, model)
    template = configs[candidate_name('equal_context', 1.)]
    for family, operator in NEW.items():
        for amplitude in AMPLITUDES:
            configs[candidate_name(family, amplitude)] = dict(template, family=family, context_operator=operator,
                                                              multiplier=amplitude)
    return configs


def gaussian_kernel(ring):
    inner, outer, far_inner, far_outer = map(int, ring.split('-'))
    x = np.arange(-(far_outer // 2), far_outer // 2 + 1, dtype=np.float64)
    radius = np.maximum(abs(x[:, None]), abs(x[None, :]))
    near = (radius > inner // 2) & (radius <= outer // 2)
    far = (radius > far_inner // 2) & (radius <= far_outer // 2)
    annulus = .62 * near / near.sum() + .38 * far / far.sum()
    target = float(np.sum(annulus * x[None, :] ** 2))
    if target >= np.mean(x ** 2):
        raise ValueError('Gaussian matching infeasible within the annulus footprint')
    low, high = .01, 1000.
    for _ in range(55):
        sigma = (low + high) / 2
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        if float(kernel @ x ** 2) < target:
            low = sigma
        else:
            high = sigma
    if abs(float(kernel @ x ** 2) - target) > 1e-9:
        raise ValueError('Matched Gaussian moment did not converge')
    return kernel.astype(np.float32), target, sigma


def score(fields, config):
    operator = config['context_operator']
    if operator not in NEW.values() or not config['multiplier']:
        return op.apply_prepared(fields, config)
    if operator not in fields:
        p, e = fields['prob'], fields['edge']
        if operator == 'matched_gaussian':
            kernel, _, _ = gaussian_kernel(config['ring'])
            ps, es = (cv2.sepFilter2D(a, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT_101) for a in (p, e))
        else:
            e = np.roll(e, (e.shape[0] // 3, e.shape[1] // 3), axis=(0, 1))
            ps, es = (op.ring_surround(a, config['ring']) for a in (p, e))
        fields[operator] = .5 * op.contrast(p, ps) + .5 * op.contrast(e, es)
    return op.sigmoid(fields['base'] + float(config['alpha']) * float(config['multiplier']) * fields['uncertainty'] * fields[operator])


def match(score_map, labels, thresholds, max_dist):
    global _MATCHERS
    if _MATCHERS is None:
        import torch
        from scripts.baselines.evaluate_official_edges import maybe_nms
        from unified_bsds_evaluator import evaluate_boundaries
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        _MATCHERS = maybe_nms, evaluate_boundaries
    nms, evaluator = _MATCHERS
    thinned = nms([score_map], device_name='cpu', low_threshold=.02)[0]
    cr, sr, cp, sp, actual = evaluator(thinned, labels, thresholds=np.asarray(thresholds),
                                      max_dist=max_dist, apply_thinning=True)
    counts = np.stack([cp, sp, cr, sr], axis=1)
    if not np.array_equal(thresholds, actual) or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError('Invalid pinned matcher outputs')
    return counts.tolist()


def checked_record(path, binding):
    if not path.exists():
        return None
    record = read(path)
    if record['binding_sha256'] != binding:
        raise ValueError(f'Existing record uses different inputs or protocol: {path}')
    return record


def source_worker(job):
    model, entry, configs, binding, max_dist = job
    sid = entry['sample_id']
    saved = CACHE / 'counts/source' / model / f'{sid}.json'
    record = checked_record(saved, binding)
    if record:
        return record
    prior = read(entry['prior_counts'])
    old_digest = prior.get('input_cache_sha256', prior.get('prediction_cache_sha256'))
    if prior['sample_id'] != sid or old_digest != entry['cache_sha256'] or sha(entry['cache_file']) != old_digest:
        raise ValueError(f'Source cache or historical count identity differs for {model}/{sid}')
    old_names = (BASE, 'plain_identity', 'legacy_main', *(candidate_name(f, m) for f in EXISTING for m in AMPLITUDES))
    old_indices = np.asarray(THRESHOLD_INDEX) - 1
    counts = {name: np.asarray(prior['candidates'][name]['counts'])[old_indices].tolist() for name in old_names}
    from PIL import Image
    with np.load(entry['cache_file'], allow_pickle=False) as cached:
        fields = op.prepared(cached['prob'], cached['edge'], configs[BASE])
        if 'gt_path' in entry:
            if sha(entry['gt_path']) != prior['gt_sha256']:
                raise ValueError(f'Source ground truth identity differs for {sid}')
            with Image.open(entry['gt_path']) as gt:
                labels = [np.asarray(gt.convert('L')) > 0]
        else:
            labels = [cached['target'] > 0]
    for family in NEW:
        for amplitude in AMPLITUDES:
            name = candidate_name(family, amplitude)
            counts[name] = match(score(fields, configs[name]), labels, THRESHOLDS, max_dist)
    record = dict(sample_id=sid, binding_sha256=binding, counts=counts, prior_counts_sha256=sha(entry['prior_counts']))
    atomic(saved, record)
    return record


def fscore(counts):
    mp, pp, mt, tt = np.moveaxis(np.asarray(counts), -1, 0)
    precision = mp / np.maximum(pp, 1)
    recall = mt / np.maximum(tt, 1)
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def source_selection(records, configs):
    sums = {name: np.sum([row['counts'][name] for row in records], axis=0) for name in configs}
    arms = {name: [name] for name in (BASE, 'plain_identity', 'legacy_main')}
    arms.update({family: [BASE, *(candidate_name(family, m) for m in AMPLITUDES)] for family in (*EXISTING, *NEW)})
    selected = {}
    for family, options in arms.items():
        candidates = [(float(fscore(sums[name][idx])), -configs[name]['multiplier'], -float(tau), name)
                      for name in options for idx, tau in enumerate(THRESHOLDS)]
        source_f, _, negative_tau, name = max(candidates)
        selected[family] = dict(config=configs[name], threshold=-negative_tau, selected_name=name,
                                F_source_development=source_f)
    ring = selected['equal_context']
    for control, alternative in (('gaussian_at_ring_setting', 'gaussian_equal_context'),
                                 ('shifted_at_ring_setting', 'shifted_sobel_equal_context')):
        name = candidate_name(alternative, ring['config']['multiplier']) if ring['config']['multiplier'] else BASE
        selected[control] = dict(config=configs[name], threshold=ring['threshold'], selected_name=name,
                                 F_source_development=None)
    return selected


def get_model(name, device):
    key = name, device
    if key in _BACKBONES:
        return _BACKBONES[key]
    import backbone_runtime as runtime
    import torch
    if name != 'edgeanchor_nyudv2':
        backbone = runtime.FrozenBackbone(name, device)
        expected = '0322caf70f588355aaaf59c2bf5872b21a4b7e9f679971a7a3bb1f69b56a01ba'
    else:
        from edge_model.models.build import build_model
        from edge_model.data.transforms import make_eval_transform
        identity = read(OLD_STAGE / 'source_cache/nyudv2/identity.json')
        backbone = runtime.FrozenBackbone.__new__(runtime.FrozenBackbone)
        backbone.name = name
        backbone.device = torch.device(device)
        backbone.config = identity['evaluation_config']
        backbone.input_size = int(backbone.config['dataset']['input_size'])
        backbone.checkpoint = runtime.WEIGHTS / 'nyudv2_strict/best.pt'
        backbone.checkpoint_sha256 = sha(backbone.checkpoint)
        expected = identity['checkpoint_sha256']
        checkpoint = torch.load(backbone.checkpoint, map_location='cpu', weights_only=False)
        backbone.model = build_model(backbone.config)
        backbone.model.load_state_dict(checkpoint['model'], strict=True)
        dataset = backbone.config['dataset']
        backbone.transform = make_eval_transform(backbone.input_size,
                                                  preserve_aspect=bool(dataset['preserve_aspect_eval']),
                                                  native_size=bool(dataset['native_size_eval']),
                                                  binarize_edges=False, gt_mode='soft', return_meta=True)
        backbone.model.requires_grad_(False).eval().to(backbone.device)
        backbone.parameters = sum(p.numel() for p in backbone.model.parameters())
    if backbone.checkpoint_sha256 != expected:
        raise ValueError(f'Unexpected checkpoint bytes: {backbone.checkpoint}')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _BACKBONES[key] = backbone
    return backbone


def verify_nyud_adapter(device):
    from backbone_runtime import read_rgb, sobel_from_rgb
    backbone = get_model('edgeanchor_nyudv2', device)
    entry = source_entries('edgeanchor_nyudv2')[0]
    with np.load(entry['cache_file'], allow_pickle=False) as stored:
        metadata = json.loads(stored['metadata_json'].item())
        filename = metadata['input_record']['image_path']
        if sha(filename) != metadata['image_sha256']:
            raise ValueError('NYUD parity image changed')
        rgb = read_rgb(filename)
        p, metadata = backbone.infer_rgb(rgb)
        e = sobel_from_rgb(rgb)
        max_p = float(np.max(np.abs(p - stored['prob'])))
        max_e = float(np.max(np.abs(e - stored['edge'])))
    if max_p > 1e-4 or max_e > 1e-6:
        raise ValueError(f'NYUD native parity gate failed: max ΔPa={max_p}, max ΔSobel={max_e}')
    return dict(sample_id=entry['sample_id'], max_abs_delta_Pa=max_p, max_abs_delta_E=max_e,
                network_shape=metadata['network_shape'], native_size_eval=True)


def target_labels(dataset, filename):
    if dataset != CID:
        from scipy.io import loadmat
        from scipy.sparse import issparse
        cells = loadmat(filename, squeeze_me=True, struct_as_record=False)['groundTruth']
        boundaries = [item.Boundaries for item in np.asarray(cells, dtype=object).reshape(-1)]
        return [np.asarray(b.toarray() if issparse(b) else b, dtype=bool) for b in boundaries]
    from PIL import Image
    with Image.open(filename) as image:
        line_drawing = np.asarray(image.convert('L'))
    median = float(np.median(line_drawing))
    mask = line_drawing < 128 if median >= 128 else line_drawing > 127
    if not 0 < mask.mean() < .3:
        raise ValueError(f'Unexpected CID line drawing foreground fraction at {filename}')
    return [mask]


def target_worker(job):
    model, device, dataset, entry, arms, binding = job
    sid = entry['sample_id']
    path = CACHE / 'counts/target' / dataset / model / f'{sid}.json'
    record = checked_record(path, binding)
    if record:
        return record
    if sha(entry['image']) != entry['image_sha256'] or sha(entry['gt']) != entry['gt_sha256']:
        raise ValueError(f'Admitted image/GT bytes changed: {dataset}/{sid}')
    map_path = CACHE / 'maps' / dataset / model / f'{sid}.npz'
    map_receipt = map_path.with_suffix('.json')
    if map_receipt.exists() and read(map_receipt)['binding_sha256'] == binding and sha(map_path) == read(map_receipt)['map_sha256']:
        with np.load(map_path, allow_pickle=False) as previous:
            prob, edge = previous['prob'], previous['edge']
    else:
        backbone = get_model(model, device)
        from backbone_runtime import read_rgb, sobel_from_rgb
        rgb = read_rgb(entry['image'])
        prob, _ = backbone.infer_rgb(rgb)
        edge = sobel_from_rgb(rgb)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = map_path.with_name(map_path.name + f'.{os.getpid()}.part')
        with temporary.open('wb') as handle:
            np.savez(handle, prob=prob, edge=edge)
        os.replace(temporary, map_path)
        atomic(map_receipt, dict(binding_sha256=binding, map_sha256=sha(map_path), backbone=backbone.manifest()))
    labels = target_labels(dataset, entry['gt'])
    if not labels or any(label.shape != prob.shape for label in labels):
        raise ValueError(f'Ground-truth dimensions differ: {dataset}/{sid} {prob.shape} {[x.shape for x in labels]}')
    fields = op.prepared(prob, edge, arms[BASE]['config'])
    memo = {}
    counts = {}
    for name, arm in arms.items():
        key = arm['selected_name'], arm['threshold']
        if key not in memo:
            memo[key] = match(score(fields, arm['config']), labels, [arm['threshold']], .0075)[0]
        counts[name] = memo[key]
    record = dict(sample_id=sid, binding_sha256=binding, counts=counts, annotations=len(labels))
    atomic(path, record)
    return record


def run_jobs(pool, worker, jobs, phase, model, dataset=None):
    total = len(jobs)
    start = time.monotonic()
    futures = {pool.submit(worker, job): job for job in jobs}
    rows = []
    state(phase=phase, model=model, dataset=dataset, completed_images=0, total_images=total, eta_seconds=None)
    for future in as_completed(futures):
        rows.append(future.result())
        n = len(rows)
        if n <= 4 or n % 5 == 0 or n == total:
            elapsed = time.monotonic() - start
            state(phase=phase, model=model, dataset=dataset, completed_images=n, total_images=total,
                  elapsed_seconds=elapsed, eta_seconds=round(elapsed / n * (total - n)))
    return sorted(rows, key=lambda row: row['sample_id'])


def summarize(records, arms, model, dataset):
    counts = {name: np.asarray([row['counts'][name] for row in records]) for name in arms}
    fvalues = {name: float(fscore(value.sum(axis=0))) for name, value in counts.items()}
    n = len(records)
    weights = np.random.default_rng(4517).multinomial(n, np.ones(n) / n, size=5000)
    draws = {name: fscore(weights @ value) for name, value in counts.items()}
    metrics = {}
    for name, arm in arms.items():
        cp, sp, cr, sr = counts[name].sum(axis=0)
        metrics[name] = dict(F_fixed=fvalues[name], precision_fixed=float(cp / max(sp, 1)),
                             recall_fixed=float(cr / max(sr, 1)), threshold=arm['threshold'],
                             multiplier=arm['config']['multiplier'])
    contrasts = []
    for first, second in COMPARE:
        low, high = np.quantile(draws[first] - draws[second], [.025, .975])
        contrasts.append(dict(first=first, second=second, delta_F=fvalues[first] - fvalues[second],
                              ci95_paired_image_low=float(low), ci95_paired_image_high=float(high)))
    result = dict(backbone=model, target=dataset, n_images=n, metrics=metrics, comparisons=contrasts,
                  interval='5000 paired whole-image resamples, unadjusted descriptive 95% intervals',
                  threshold_policy='Every arm uses its source-development-fixed threshold; no target tuning',
                  target_ap_computed=False)
    atomic(OUT / dataset / f'{model}.json', result)
    csv_path = OUT / dataset / f'{model}_comparisons.csv'
    temp = csv_path.with_suffix('.csv.part')
    with temp.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=contrasts[0].keys())
        writer.writeheader()
        writer.writerows(contrasts)
    os.replace(temp, csv_path)
    return result


def prepare_targets():
    target_path = OUT / 'target_admission.json'
    if target_path.exists():
        return read(target_path)
    bsds_archive = OLD_CACHE / 'downloads/BSR_bsds500.tgz'
    cid_archive = CACHE / 'incoming/CID_contours.tar.gz'
    groups = {BSDS: berkeley(bsds_archive, CACHE / 'datasets/bsds_train')}
    cid = groningen(cid_archive, CACHE / 'datasets/cid')
    if cid:
        groups[CID] = cid
    official = OLD_STAGE / 'asset_staging/data_d1c9dd8694dc/RBCM-Edge-Data/edge_data/official_rbcm'
    audit = adjudicate(groups, prior_images(official, OLD_CACHE))
    audit['sources'] = {
        BSDS: dict(url='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html',
                   archive_sha256=sha(bsds_archive), scope='Distinct previously unscored project images within BSDS; official TRAIN split, not benchmark test or a separate corpus.'),
        CID: dict(url='https://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html',
                  archive_sha256=sha(cid_archive) if cid else None,
                  scope='Independent corpus with hand-drawn contours; distinct GT conventions, report separately.',
                  acquisition='admitted with overlap checks' if cid else 'archive unavailable; no CID metrics')}
    frozen(target_path, audit)
    return audit


def run(args):
    state(phase='verify_and_freeze_protocol', workers=args.workers, device=args.device)
    matcher_path = OLD_OUT / 'evaluator/matcher_config.json'
    matcher = read(matcher_path)
    for path, digest in matcher['file_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Pinned matcher file has changed: {path}')
    entries = {name: source_entries(name) for name in MODELS}
    configs = {name: configurations(name) for name in MODELS}
    kernel, variance, sigma = gaussian_kernel(configs['edgeanchor_nyudv2'][BASE]['ring'])
    protocol = dict(phase='locked_before_target_metrics', code_sha256=sha(__file__),
                    admission_code_sha256=sha(Path(__file__).with_name('admission.py')),
                    matcher_sha256=sha(matcher_path), operator_sha256=sha(op.__file__),
                    frozen_adapter_sha256=sha(ROUND7 / 'backbone/backbone_runtime.py'),
                    source_entries=entries, candidate_configs=configs, thresholds=THRESHOLDS.tolist(),
                    family_selection='Common source images and three amplitude choices (0/.5/1); max pooled F, tie lower amplitude then lower threshold.',
                    previous_development_exposure=True, source_selection_independent_of_target=True,
                    fixed_tau_primary=True, device=args.device, comparisons=COMPARE,
                    gaussian={'kernel': kernel.tolist(), 'second_moment_px_squared': variance, 'sigma_px': sigma},
                    translated_null='Translate entire Sobel map and its surround by one-third image height/width with toroidal wrap. Same pixels, histogram and most local structure; alignment and seam differ.',
                    bootstrap='Seed 4517, paired whole-image bootstrap 5000, report raw 95% intervals separately by corpus and checkpoint')
    # JSON serializes the comparison tuples as lists; normalize now for equality on restart.
    protocol = json.loads(json.dumps(protocol))
    source_protocol_path = OUT / 'protocol.json'
    if source_protocol_path.exists() and read(source_protocol_path)['code_sha256'] == ORIGINAL_SOURCE_RUNNER_SHA:
        # Preserve source receipts. Only the native-size target adapter and target audit binding changed.
        protocol['code_sha256'] = ORIGINAL_SOURCE_RUNNER_SHA
    protocol_sha = frozen(source_protocol_path, protocol)
    target_runtime = dict(runner_sha256=sha(__file__), original_source_runner_sha256=ORIGINAL_SOURCE_RUNNER_SHA,
                          source_selection_implementation_unchanged=True, device=args.device,
                          NYUD_native_parity=verify_nyud_adapter(args.device))
    target_runtime_sha = frozen(OUT / 'target_runtime.json', target_runtime)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        selected = {}
        for model, source in MODELS.items():
            jobs = [(model, entry, configs[model], protocol_sha, matcher['source_max_dist'][source]) for entry in entries[model]]
            records = run_jobs(pool, source_worker, jobs, 'source_development', model)
            selected[model] = source_selection(records, configs[model])
        source_sha = frozen(OUT / 'source_frozen.json', dict(protocol_sha256=protocol_sha, models=selected))
        state(phase='admit_targets', frozen_source_sha256=source_sha)
        admission = prepare_targets()
        reports = []
        binding = hashlib.sha256((source_sha + target_runtime_sha + sha(OUT / 'target_admission.json')).encode()).hexdigest()
        # Score the independent corpus first whenever its archive is available.
        for dataset in (CID, BSDS):
            if dataset not in admission['datasets']:
                continue
            target = admission['datasets'][dataset]['entries']
            if not target:
                continue
            for model in MODELS:
                jobs = [(model, args.device, dataset, entry, selected[model], binding) for entry in target]
                records = run_jobs(pool, target_worker, jobs, 'fixed_tau_target', model, dataset)
                reports.append(summarize(records, selected[model], model, dataset))
    atomic(OUT / 'results.json', dict(frozen_source_sha256=source_sha, target_admission=admission['sources'], reports=reports))
    state(status='completed', phase='summarized', results_path=str(OUT / 'results.json'),
          completed_reports=len(reports), eta_seconds=0, cid_scored=any(r['target'] == CID for r in reports))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.device != 'cpu' and args.workers != 1:
        parser.error('Use one worker when assigning this runner to a GPU')
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Engineering runner already owns the lock', flush=True)
            return
        try:
            run(args)
        except Exception as error:
            state(status='failed', phase='error', error=repr(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
