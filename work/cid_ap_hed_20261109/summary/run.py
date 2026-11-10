"""Publish the common CID comparison after both independent workers finish."""

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/RBCM_CID_AP_HED_20261109'
PREVIOUS = ROOT / 'outputs/RBCM_validation_20261108/engineering'
STATUS = OUT / 'summary/status.json'
REPORT = OUT / 'CID结果汇总.txt'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def write(path, data):
    temp = path.with_name(path.name + '.part')
    temp.write_text(data, encoding='utf-8')
    temp.replace(path)


def status(state, phase, **details):
    write(STATUS, json.dumps(dict(status=state, phase=phase, **details), ensure_ascii=False, indent=2))


def wait_for_results():
    while True:
        branches = {item['id']: item for item in read(OUT / 'status.json').get('branches', [])}
        states = []
        for name in ('ap', 'hed'):
            local = read(OUT / name / 'status.json')
            states.append(local.get('status') if local.get('status') in {'complete', 'completed', 'failed', 'blocked'}
                          else branches.get(name, {}).get('state', 'waiting'))
        if all(value in {'complete', 'completed'} for value in states):
            return True
        if any(value in {'failed', 'blocked'} for value in states):
            status('blocked', '上游分支未完成', upstream=dict(zip(('ap', 'hed'), states)))
            return False
        time.sleep(10)


def summarize():
    ap = read(OUT / 'ap/results.json')
    hed = read(OUT / 'hed/results.json')
    hed_protocol = read(OUT / 'hed/protocol.json')
    admission = read(PREVIOUS / 'target_admission.json')['datasets']['groningen_cid']['entries']
    admitted = {entry['sample_id'] for entry in admission}
    if (set(hed['target_sample_ids']) != admitted or hed['n_images'] != 38 or len(admitted) != 38
            or ap['target'] != 'groningen_cid' or hed['dataset'] != ap['target']
            or hed_protocol['metrics_sha256'] != ap['shared_metric_sha256']):
        raise ValueError('CID membership, dataset or evaluator mismatch')
    lines = ['Groningen CID · 38 张图 · 两种冻结骨干与作者发布的 HED',
             'AP、ODS、OIS 和部署 F 均以百分数表示。部署阈值在源开发集锁定。',
             'ODS/OIS 为 CID 阈值扫描统计。AP 沿用 Round 7 BSDS 101 点口径。', '',
             '骨干\t分支\tAP\tODS\tOIS\t源阈值\t部署 F']

    def row(model, arm, metric):
        if metric['n_images'] != 38 or metric['thresholds'] != hed['metrics']['thresholds']:
            raise ValueError(f'Unexpected image count or threshold grid for {model} / {arm}')
        return '\t'.join([model, arm, *(f'{metric[key] * 100:.3f}' for key in ('AP_BSDS_101point', 'ODS', 'OIS')),
                          f"{metric['fixed_development_threshold']:.2f}", f"{metric['F_fixed'] * 100:.3f}"])

    lines.append(row(hed['backbone'], hed['arm'], hed['metrics']))
    for model, report in ap['reports'].items():
        lines.extend(row(model, arm, metric) for arm, metric in report['arms'].items())
    lines.extend(['', '配对 AP 差值｜百分点｜前者减后者｜95% 逐图配对 bootstrap 区间',
                  '骨干\t前者\t后者\tΔAP\t95% 下界\t95% 上界'])
    for model, report in ap['reports'].items():
        for pair in report['comparisons']:
            lines.append('\t'.join([model, pair['first'], pair['second'],
                                   *(f'{pair[key] * 100:+.3f}' for key in ('delta_AP', 'ci95_paired_image_low',
                                                                        'ci95_paired_image_high'))]))
    write(REPORT, '\n'.join(lines) + '\n')


if __name__ == '__main__':
    status('running', '等待两支路完整统计')
    try:
        if wait_for_results():
            summarize()
            status('complete', '统一对照表和 AP 配对区间已落盘', completed=1, total=1, report=str(REPORT))
    except Exception as error:
        status('failed', '汇总未通过一致性核对', error=repr(error))
        raise
