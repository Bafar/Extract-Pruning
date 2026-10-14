"""Ordered JSONL inference, one engine per run, resumable fail-fast output."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
from pipeline import ExtractPruning, FIELDS, normalize_input


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(config):
    cfg = json.loads(config.read_text(encoding='utf-8'))
    root = config.parent
    files = {config.resolve(), *(Path(__file__).parent/name for name in
                               ('pipeline.py','gate_runtime.py','user_runtime.py','batch_predict.py'))}
    for side in ('user','assistant'):
        files.add((root/cfg[side]['prompt']).resolve())
        files.update(p.resolve() for p in (root/cfg[side]['adapter']).rglob('*') if p.is_file())
    files.update(p.resolve() for p in (root/cfg['base_model']).rglob('*') if p.is_file())
    files.update(p.resolve() for p in (root/cfg['gate']['weights']).parent.rglob('*') if p.is_file())
    return {str(p):digest(p) for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.input, args.output, args.config = (p.resolve() for p in (args.input,args.output,args.config))
    if args.input == args.output: raise ValueError('不能覆盖输入')
    manifest = args.output.with_suffix('.run.json')
    diagnostic_path = args.output.with_suffix('.diagnostics.jsonl')
    if args.input in (manifest,diagnostic_path): raise ValueError('输出辅助文件不能覆盖输入')
    rows = []
    with args.input.open(encoding='utf-8-sig') as stream:
        for number,line in enumerate(stream,1):
            if not line.strip(): raise ValueError(f'输入第{number}行为空；为保持行对应不跳过空行')
            rows.append(normalize_input(json.loads(line)))
    if not rows: raise ValueError('空输入文件')
    signature = {'input_sha256':digest(args.input), 'files':fingerprint(args.config)}
    completed = 0
    if args.resume:
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['signature'] != signature: raise ValueError('输入、配置、代码或模型改变，不能恢复旧结果')
        raw = args.output.read_bytes()
        if raw and not raw.endswith(b'\n'): raise ValueError('最后一行写入不完整，请检查后再恢复')
        records = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if len(records)>len(rows) or any(not isinstance(r,dict) or set(r)!=set(FIELDS) for r in records):
            raise ValueError('已有输出行数或字段结构无效')
        completed = len(records)
        if saved.get('output_sha256') and saved['output_sha256'] != digest(args.output):
            raise ValueError('已完成结果被修改')
    elif any(p.exists() for p in (args.output, manifest, diagnostic_path)):
        raise FileExistsError('结果已存在，请使用 --resume 或新的 --output 路径')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    state = {'signature':signature,'input':str(args.input),'output':str(args.output),
             'total':len(rows),'completed':completed,'status':'running'}
    def save_state():
        temp=manifest.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(manifest)
    save_state()
    started=time.monotonic()
    engine=ExtractPruning(args.config)
    with args.output.open('a' if args.resume else 'w',encoding='utf-8',newline='\n') as out, diagnostic_path.open('a' if args.resume else 'w',encoding='utf-8',newline='\n') as diag:
        for index in range(completed,len(rows)):
            step_start=time.monotonic()
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result, details=engine.predict_with_diagnostics(rows[index])
                out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush()
                diag.write(json.dumps({'line':index+1,'seconds':time.monotonic()-step_start,'diagnostics':details},ensure_ascii=False)+'\n'); diag.flush()
                state['completed']=index+1
                save_state()
                print(f'[{index+1}/{len(rows)}] {time.monotonic()-step_start:.1f}s',flush=True)
            except BaseException as exc:
                state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', failed_line=index+1,
                             error=f'{type(exc).__name__}: {exc}')
                save_state()
                raise
    state.update(status='complete', completed=len(rows), elapsed_seconds_this_run=time.monotonic()-started,
                 output_sha256=digest(args.output))
    save_state()
    print(f'完成：{len(rows)}条，结果：{args.output}',flush=True)


if __name__ == '__main__':
    main()
