"""Run source selection, CID evaluation, and latency in dependency order."""

import ctypes
from datetime import datetime, timedelta, timezone
from html import escape
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer


WORK = Path(__file__).resolve().parent
OUT = WORK.parents[1] / 'outputs/RBCM_HED_AUG_20261110'
PORT = 8785
TIMEZONE = timezone(timedelta(hours=8))
TERMINAL = {'complete', 'completed', 'failed', 'blocked'}
PASS = {'complete', 'completed'}

spec = importlib.util.spec_from_file_location('cid_supervisor_helpers', WORK.parent / 'cid_ap_hed_20261109/supervise.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
helpers.WORK = WORK


def now():
    return datetime.now(TIMEZONE).isoformat(timespec='seconds')


def render(items, stage):
    names = {'waiting': '等待中', 'running': '运行中', 'complete': '已完成',
             'completed': '已完成', 'failed': '失败', 'blocked': '受阻'}
    stamp = now()
    lines = [f'HED 接入 RBCM · {stamp}', f'总状态：{stage}', '']
    table = []
    for item in items:
        state = item['state']
        raw = item['details']
        done = raw.get('completed', raw.get('completed_images', raw.get('done')))
        total = raw.get('total', raw.get('total_images'))
        count = f'{done}/{total}' if done is not None and total is not None else '—'
        eta = raw.get('eta_seconds')
        remaining = f'{eta / 60:.1f} 分钟（当前阶段）' if isinstance(eta, (int, float)) and eta > 0 else item['initial_eta']
        fields = [item['label'], names.get(state, state), str(raw.get('phase') or item['pending']), count,
                  remaining if state not in TERMINAL else '—', str(raw.get('error') or item.get('error', ''))]
        lines.append('｜'.join(field for field in fields if field))
        table.append('<tr>' + ''.join(f'<td>{escape(field)}</td>' for field in fields) + '</tr>')
    lines.extend(['', '顺序：源域冻结 → CID 匹配与配对统计 → 端到端计时。',
                  'ODS/OIS 单列，部署 F 使用源域锁定阈值。'])
    page = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="refresh" content="15"><title>HED 接入 RBCM</title><style>body{{font:16px system-ui,sans-serif;margin:3rem auto;max-width:1100px;padding:0 1rem;background:#f5f8fa;color:#142c3d}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dde5ec}}th{{background:#e5eef6}}</style>'
            f'<h1>HED 接入 RBCM</h1><p>{stamp} · {escape(stage)} · 每 15 秒刷新</p><table><thead><tr><th>分支</th><th>状态</th><th>阶段</th><th>完成</th><th>预计剩余</th><th>信息</th></tr></thead>'
            f'<tbody>{"".join(table)}</tbody></table><p><a href="STATUS.txt">文字进度</a>　<a href="status.json">详细状态</a>　<a href="validation_plan.txt">验证方案</a></p></html>')
    data = json.dumps(dict(status=stage, updated_at=stamp, pid=os.getpid(), branches=items), ensure_ascii=False, indent=2)
    for url, filename, value, kind in [('/', '实时进度.html', page, 'text/html'),
                                       ('/STATUS.txt', 'STATUS.txt', '\n'.join(lines) + '\n', 'text/plain'),
                                       ('/status.json', 'status.json', data, 'application/json')]:
        helpers.VIEWS[url] = (value.encode('utf-8'), kind)
        helpers.publish(OUT / filename, value)
    plan = OUT / 'validation_plan.txt'
    if plan.exists():
        helpers.VIEWS['/validation_plan.txt'] = (plan.read_bytes(), 'text/plain')


def watch():
    try:
        server = ThreadingHTTPServer(('127.0.0.1', PORT), helpers.StatusHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except OSError as error:
        helpers.publish(OUT / 'dashboard_error.txt', str(error))
    processes, starts = {}, {}
    while True:
        items, states = [], {}
        for job in helpers.read(OUT / 'jobs.json').get('jobs', []):
            ident = job['id']
            folder = OUT / ident
            folder.mkdir(exist_ok=True)
            raw = helpers.read(folder / 'status.json')
            state = str(raw.get('status', raw.get('state', 'waiting'))).lower()
            dependencies = {key: states.get(key, 'waiting') for key in job.get('requires', [])}
            error = ''
            if state not in TERMINAL:
                if any(value in {'failed', 'blocked'} for value in dependencies.values()):
                    state, error = 'blocked', '上游分支未通过'
                elif job['enabled'] and all(value in PASS for value in dependencies.values()):
                    try:
                        script = helpers.entry(job)
                        if ident in processes:
                            child, adopted = processes[ident]
                            if (child is not None and child.poll() is not None) or (adopted and helpers.existing(script) != adopted):
                                del processes[ident]
                        if ident not in processes:
                            if starts.get(ident, 0) >= 2:
                                raise RuntimeError('进程连续退出，请检查本分支 stderr.log')
                            child, adopted = processes[ident] = helpers.start(job, folder, script)
                            starts[ident] = starts.get(ident, 0) + 1
                            receipt = dict(launched=now(), windows_pid=child.pid if child else None,
                                           adopted_wsl_pid=adopted, attempt=starts[ident], args=job.get('args', []))
                            helpers.publish(folder / 'launch_receipt.json', json.dumps(receipt, indent=2))
                        state = 'running'
                    except (OSError, RuntimeError, subprocess.SubprocessError) as exception:
                        error, state = str(exception), 'failed'
                else:
                    state = 'waiting'
            states[ident] = state
            items.append(dict(id=ident, label=job['label'], state=state, details=raw, error=error,
                              pending=job['pending'], initial_eta=job['initial_eta']))
        ended = bool(items) and all(value in TERMINAL for value in states.values())
        render(items, '已完成' if ended and all(value in PASS for value in states.values()) else '已结束，请查分支' if ended else '进行中')
        if ended:
            return
        time.sleep(10)


if __name__ == '__main__':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, 'Global\\RBCM_HED_AUG_20261110')
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if ctypes.get_last_error() != 183:
            watch()
    finally:
        kernel.CloseHandle(handle)
