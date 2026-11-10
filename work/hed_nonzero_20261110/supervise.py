"""Supervise the source-locked HED nonzero sensitivity run on Windows."""

import ctypes
from html import escape
import importlib.util
import json
import os
from pathlib import Path


WORK = Path(__file__).resolve().parent
OUT = WORK.parents[1] / 'outputs/RBCM_HED_NONZERO_20261110'
spec = importlib.util.spec_from_file_location('hed_nonzero_watch', WORK.parent / 'hed_rbcm_20261110/supervise.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.WORK = WORK
runner.OUT = OUT
runner.PORT = 8786
runner.helpers.WORK = WORK


def render(items, stage):
    stamp = runner.now()
    names = {'waiting': '等待中', 'running': '运行中', 'complete': '已完成',
             'completed': '已完成', 'failed': '失败', 'blocked': '受阻'}
    lines = [f'HED 非零幅度敏感性分析 · {stamp}', f'总状态：{stage}', '']
    rows = []
    for item in items:
        state, raw = item['state'], item['details']
        done = raw.get('completed', raw.get('done'))
        total = raw.get('total')
        count = f'{done}/{total}' if done is not None and total is not None else '—'
        eta = raw.get('eta_seconds')
        if state in runner.TERMINAL:
            time = '—'
        elif isinstance(eta, (int, float)) and eta > 0:
            time = f'{eta / 60:.1f} 分钟（当前阶段）'
        else:
            time = item['initial_eta']
        fields = (item['label'], names.get(state, state), str(raw.get('phase') or item['pending']), count,
                  time, str(raw.get('error') or item.get('error', '')))
        lines.append('｜'.join(field for field in fields if field))
        rows.append('<tr>' + ''.join(f'<td>{escape(field)}</td>' for field in fields) + '</tr>')
    lines.extend(['', '各上下文分支仅在 BIPED 源域从 0.5、1 选择幅度，锁定部署阈值。',
                  'CID 报告 AP、部署 F 和 95% 逐图配对区间。'])
    page = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="refresh" content="15"><title>HED 非零幅度敏感性分析</title><style>body{{font:16px system-ui,sans-serif;margin:3rem auto;max-width:1050px;padding:0 1rem;background:#f5f8fa;color:#142c3d}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dde5ec}}th{{background:#e5eef6}}</style>'
            f'<h1>HED 非零幅度敏感性分析</h1><p>{stamp} · {escape(stage)} · 每 15 秒刷新</p><table><thead><tr><th>分支</th><th>状态</th><th>阶段</th><th>完成</th><th>预计剩余</th><th>信息</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table><p><a href="STATUS.txt">文字进度</a>　<a href="status.json">详细状态</a>　<a href="validation_plan.txt">验证方案</a></p></html>')
    data = json.dumps(dict(status=stage, updated_at=stamp, pid=os.getpid(), branches=items), ensure_ascii=False, indent=2)
    for url, filename, payload, kind in [('/', '实时进度.html', page, 'text/html'),
                                         ('/status.json', 'status.json', data, 'application/json'),
                                         ('/STATUS.txt', 'STATUS.txt', '\n'.join(lines) + '\n', 'text/plain')]:
        runner.helpers.VIEWS[url] = (payload.encode('utf-8'), kind)
        runner.helpers.publish(OUT / filename, payload)
    runner.helpers.VIEWS['/validation_plan.txt'] = ((OUT / 'validation_plan.txt').read_bytes(), 'text/plain')


if __name__ == '__main__':
    runner.render = render
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, 'Global\\RBCM_HED_NONZERO_20261110')
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if ctypes.get_last_error() != 183:
            runner.watch()
    finally:
        kernel.CloseHandle(handle)
