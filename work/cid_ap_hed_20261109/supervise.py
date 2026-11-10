"""Launch resumable CID workers from Windows and expose their status on loopback."""

import ctypes
import datetime as dt
import html
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


WORK = Path(__file__).resolve().parent
OUT = WORK.parents[1] / 'outputs/RBCM_CID_AP_HED_20261109'
ZONE = dt.timezone(dt.timedelta(hours=8))
PORT = 8784
TERMINAL = {'complete', 'completed', 'failed', 'blocked'}
WSL = ['wsl.exe', '-d', 'Ubuntu-22.04', '--']
VIEWS = {}


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def publish(path, data):
    temp = path.with_name(f'.{path.name}.{os.getpid()}.part')
    try:
        temp.write_text(data, encoding='utf-8')
        temp.replace(path)
    except OSError:
        pass  # File viewers can lock a snapshot without interrupting the jobs.


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        view = VIEWS.get(path)
        if view is None:
            self.send_error(404)
            return
        payload, kind = view
        self.send_response(200)
        self.send_header('Content-Type', f'{kind}; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def render(items, stage):
    stamp = dt.datetime.now(ZONE).isoformat(timespec='seconds')
    data = json.dumps(dict(status=stage, updated_at=stamp, pid=os.getpid(), branches=items),
                      ensure_ascii=False, indent=2)
    lines = [f'CID 完整 AP 与标准 HED · {stamp}', f'总状态：{stage}', '']
    table = []
    names = {'waiting': '准备中', 'running': '运行中', 'complete': '已完成',
             'completed': '已完成', 'failed': '失败', 'blocked': '受阻'}
    for item in items:
        raw = item['details']
        phase = str(raw.get('phase') or item['pending'])
        done = raw.get('completed', raw.get('done', raw.get('completed_images')))
        total = raw.get('total', raw.get('total_images'))
        count = f'{done}/{total}' if done is not None and total is not None else '—'
        eta = raw.get('eta_seconds')
        eta_label = '—'
        if item['state'] not in TERMINAL:
            eta_label = item['initial_eta']
            if isinstance(eta, (int, float)) and eta > 0:
                eta_label = f'{eta / 60:.1f} 分钟（当前阶段）'
        note = str(raw.get('error') or item.get('error') or '')
        fields = [item['label'], names.get(item['state'], item['state']), phase, count, eta_label, note]
        lines.append('｜'.join(field for field in fields if field))
        table.append('<tr>' + ''.join(f'<td>{html.escape(field)}</td>' for field in fields) + '</tr>')
    lines.extend(['', 'ODS/OIS 使用目标集阈值扫描。部署 F 另用源域锁定阈值。',
                  '断点按图像或得分图保存，Windows 开机后接续。'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15"><title>CID 完整 AP 与标准 HED</title>
<style>body{font:16px system-ui,sans-serif;margin:3rem auto;max-width:1100px;padding:0 1rem;background:#f6f8fa;color:#172d3a}table{border-collapse:collapse;background:white;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #dde5ec}th{background:#e7eff5}p{line-height:1.6}</style>'''
    page += (f'<h1>CID 完整 AP 与标准 HED</h1><p>{html.escape(stamp)} · {html.escape(stage)} · 每 15 秒刷新</p>'
             '<table><thead><tr><th>分支</th><th>状态</th><th>当前阶段</th><th>已完成</th><th>估计剩余</th><th>信息</th></tr></thead>'
             f'<tbody>{"".join(table)}</tbody></table><p><a href="STATUS.txt">文字进度</a>　<a href="status.json">详细状态</a></p>'
             '<p>目标集 ODS/OIS 与源域锁定阈值的部署 F 分开报告。</p></html>')
    for url, filename, payload, kind in [('/status.json', 'status.json', data, 'application/json'),
                                         ('/STATUS.txt', 'STATUS.txt', '\n'.join(lines) + '\n', 'text/plain'),
                                         ('/', '实时进度.html', page, 'text/html')]:
        VIEWS[url] = (payload.encode('utf-8'), kind)
        publish(OUT / filename, payload)


def entry(job):
    path = (WORK / job['script']).resolve()
    if not path.is_relative_to(WORK) or not path.is_file():
        raise FileNotFoundError(path)
    return '/mnt/d/' + path.as_posix()[3:]


def existing(script):
    result = subprocess.run(WSL + ['ps', '-eo', 'pid,args'], capture_output=True, encoding='utf-8',
                            errors='replace', timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    result.check_returncode()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and 'python' in parts[1] and script in parts[2:]:
            return int(parts[0])
    return None


def start(job, directory, script):
    old = existing(script)
    if old is not None:
        return None, old
    argv = WSL + ['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                  'MKL_NUM_THREADS=1', *job.get('env', []), 'python3', '-u', script, *job.get('args', [])]
    with (directory / 'stdout.log').open('ab') as stdout, (directory / 'stderr.log').open('ab') as stderr:
        child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    return child, None


def main():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, 'Global\\RBCM_CID_AP_HED_20261109')
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if ctypes.get_last_error() == 183:
            return
        try:
            server = ThreadingHTTPServer(('127.0.0.1', PORT), StatusHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except OSError as error:
            publish(OUT / 'dashboard_error.txt', str(error))
        children, attempts = {}, {}
        while True:
            items = []
            for job in read(OUT / 'jobs.json').get('jobs', []):
                ident = job['id']
                directory = OUT / ident
                directory.mkdir(exist_ok=True)
                raw = read(directory / 'status.json')
                state = str(raw.get('status', raw.get('state', 'waiting'))).lower()
                error = ''
                if state not in TERMINAL and job['enabled']:
                    try:
                        script = entry(job)
                        if ident in children:
                            child, adopted = children[ident]
                            if (child is not None and child.poll() is not None) or (adopted and existing(script) != adopted):
                                del children[ident]
                        if ident not in children:
                            if attempts.get(ident, 0) >= 2:
                                raise RuntimeError('Worker exited twice; inspect its status and stderr.log')
                            children[ident] = start(job, directory, script)
                            attempts[ident] = attempts.get(ident, 0) + 1
                            child, adopted = children[ident]
                            receipt = dict(time=dt.datetime.now(ZONE).isoformat(timespec='seconds'),
                                           windows_pid=child.pid if child else None, adopted_wsl_pid=adopted,
                                           args=job.get('args', []), attempt=attempts[ident])
                            publish(directory / 'launch_receipt.json', json.dumps(receipt, indent=2))
                        state = 'running'
                    except (OSError, RuntimeError, subprocess.SubprocessError) as exception:
                        error, state = str(exception), 'failed'
                items.append(dict(id=ident, label=job['label'], state=state, details=raw, error=error,
                                  pending=job['pending'], initial_eta=job['initial_eta']))
            ended = bool(items) and all(item['state'] in TERMINAL for item in items)
            succeeded = ended and all(item['state'] in {'complete', 'completed'} for item in items)
            render(items, '已完成' if succeeded else '已结束，请查看分支状态' if ended else '进行中')
            if ended:
                return
            time.sleep(10)
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    main()
