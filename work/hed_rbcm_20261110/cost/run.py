"""Time source-frozen HED pipelines after all scoring workers have finished."""

import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/RBCM_HED_AUG_20261110'
COST = OUT / 'cost'
SOURCE = OUT / 'source/source_frozen.json'
ADMISSION = ROOT / 'outputs/RBCM_validation_20261108/engineering/target_admission.json'
HED = ROOT / 'work/cid_ap_hed_20261109/hed/run.py'
ARMS = ('plain_identity', 'equal_context', 'smoothing_equal_context')
FIELDS = ('inference_ms', 'context_ms', 'nms_and_threshold_ms', 'total_ms')
REPEATS = 3


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, payload):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def status(state, phase, **fields):
    save(COST / 'status.json', dict(status=state, phase=phase, **fields))


def timed(path, arm, entry, hed, prior, read_rgb, sobel_from_rgb, nms):
    start = time.perf_counter()
    probability = hed.infer(path)
    forward = time.perf_counter()
    if arm == 'plain_identity':
        score = probability
    else:
        config = entry['config']
        sobel = sobel_from_rgb(read_rgb(path)) if config['multiplier'] else np.zeros_like(probability)
        score = prior.score(prior.op.prepared(probability, sobel, config), config)
    context = time.perf_counter()
    suppressed = nms([score], device_name='cpu', low_threshold=.02)[0]
    binary = (suppressed >= entry['threshold']).astype(np.uint8)
    end = time.perf_counter()
    if binary.shape != probability.shape:
        raise ValueError(f'Unexpected output geometry for {arm}')
    return dict(zip(FIELDS, ((forward-start)*1000, (context-forward)*1000,
                             (end-context)*1000, (end-start)*1000)))


def main():
    spec = importlib.util.spec_from_file_location('hed_cost_author_adapter', HED)
    hed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hed)
    import run as prior  # The HED adapter pins the original evaluator on sys.path.
    from backbone_runtime import read_rgb, sobel_from_rgb
    from scripts.baselines.evaluate_official_edges import maybe_nms
    import torch

    torch.set_num_threads(1)
    hed.verify_assets()
    frozen = load(SOURCE)
    if frozen['target_score_accessed_for_selection'] is not False:
        raise ValueError('Expected source-only frozen settings')
    selected = frozen['selected']
    entries = load(ADMISSION)['datasets']['groningen_cid']['entries']
    sizes = []
    for entry in entries:
        with Image.open(entry['image']) as image:
            width, height = image.size
        sizes.append(dict(entry, width=width, height=height))
    sizes.sort(key=lambda entry: (entry['width'] * entry['height'], entry['sample_id']))
    images = [sizes[round(q * (len(sizes)-1))] for q in (.15, .40, .65, .90)]
    protocol = dict(arms={arm: selected[arm] for arm in ARMS}, samples=images, repeats=REPEATS,
                    runner_sha256=hed.digest(__file__), frozen_source_sha256=hed.digest(SOURCE),
                    batch=1, cpu_threads=1, platform=platform.platform(),
                    timing_scope='Includes warm-file decode, native HED forward, per-arm processing and NMS+threshold; excludes GT and matching.')
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save(COST / 'protocol.json', protocol)
    collected = []
    start = time.monotonic()
    for i, image in enumerate(images):
        path = Path(image['image'])
        if hed.digest(path) != image['image_sha256']:
            raise ValueError('CID image hash changed')
        receipt = COST / f"timing_{image['sample_id']}.json"
        if receipt.exists():
            previous = load(receipt)
            if previous['binding_sha256'] != key:
                raise ValueError('Cost receipt is bound to a different protocol')
            collected.extend(previous['rows'])
        else:
            rows = []
            for repeat in range(-1, REPEATS):
                for offset in range(len(ARMS)):
                    arm = ARMS[(i + repeat + offset) % len(ARMS)]
                    elapsed = timed(path, arm, selected[arm], hed, prior, read_rgb, sobel_from_rgb, maybe_nms)
                    if repeat >= 0:
                        rows.append(dict(sample_id=image['sample_id'], arm=arm, repeat=repeat, **elapsed))
            save(receipt, dict(binding_sha256=key, rows=rows))
            collected.extend(rows)
        status('running', '原尺寸逐图计时', completed=i+1, total=len(images),
               eta_seconds=round((time.monotonic()-start)/(i+1)*(len(images)-i-1)))
    report = {}
    for arm in ARMS:
        rows = [row for row in collected if row['arm'] == arm]
        report[arm] = {field: statistics.median(row[field] for row in rows) for field in FIELDS}
    save(COST / 'results.json', dict(protocol=protocol, median_ms=report, rows=collected))
    lines = ['标准 HED · CID 部署延迟｜CPU 单线程、批量 1，单位 ms',
             '4 个预选尺寸，每图每分支预热后重复 3 次。包含文件解码和 NMS/阈值，排除 GT 与评价匹配。', '',
             '分支\tHED 含解码\t上下文\tNMS 与阈值\t端到端']
    for arm in ARMS:
        lines.append('\t'.join([arm, *(f'{report[arm][field]:.3f}' for field in FIELDS)]))
    temporary = COST / '部署计时.txt.part'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(COST / '部署计时.txt')
    status('complete', '端到端计时已落盘', completed=len(images), total=len(images), eta_seconds=0)


if __name__ == '__main__':
    COST.mkdir(parents=True, exist_ok=True)
    with (COST / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('running', '载入源冻结配置', completed=0, total=4)
        try:
            main()
        except Exception as error:
            status('failed', '计时未完成', error=repr(error))
            raise
