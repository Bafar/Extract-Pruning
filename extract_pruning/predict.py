"""JSON file/stdin -> exactly five fields. Logs go to stderr."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from pipeline import ExtractPruning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--input', type=Path, help='UTF-8 JSON 文件，省略时从 stdin 读取')
    parser.add_argument('--output', type=Path, help='输出 JSON 文件；省略时打印到 stdout')
    parser.add_argument('--diagnostics', type=Path, help='可选诊断文件，不混入五字段结果')
    args = parser.parse_args()
    try:
        if args.output and args.input and args.output.resolve() == args.input.resolve():
            raise ValueError('输出不能覆盖输入')
        if args.diagnostics and args.diagnostics.resolve() in {p.resolve() for p in (args.input, args.output) if p}:
            raise ValueError('diagnostics 不能覆盖输入或输出')
        payload = json.loads(args.input.read_text(encoding='utf-8-sig') if args.input else sys.stdin.read())
        with contextlib.redirect_stdout(sys.stderr):
            result, diagnostics = ExtractPruning(args.config).predict_with_diagnostics(payload)
        text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_suffix(args.output.suffix + '.tmp')
            temp.write_text(text, encoding='utf-8')
            temp.replace(args.output)
        else:
            print(text, end='')
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
