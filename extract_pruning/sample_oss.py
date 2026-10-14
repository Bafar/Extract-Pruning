"""Extract reproducible unique inference inputs; never copy OSS labels into inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from pipeline import normalize_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('input_batch.jsonl'))
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error('count must be positive')
    if args.output.exists() or args.output.with_suffix('.sources.json').exists():
        raise FileExistsError('抽样输出已存在，请使用新的输出路径')
    candidates, seen, counts = [], set(), Counter()
    with args.source.open(encoding='utf-8-sig') as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip(): continue
            record = json.loads(raw)
            for position, item in enumerate(record.get('payload', []), 1):
                if not isinstance(item, dict) or item.get('source') != 'llm': continue
                if item.get('type') not in ('user_pruning', 'assistant_pruning'): continue
                counts['eligible_records'] += 1
                try: value = normalize_input(item.get('input'))
                except ValueError:
                    counts['invalid_input'] += 1
                    continue
                if not any(m['msg'] for m in value['msgs']):
                    counts['empty_input'] += 1
                    continue
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    counts['duplicate_input'] += 1
                    continue
                seen.add(key)
                candidates.append((value, {'source_line':line, 'payload_index':position,
                                           'oss_key':record.get('_oss_key'), 'type':item.get('type')}))
    if len(candidates) < args.count:
        raise ValueError(f'只有 {len(candidates)} 条独立输入，不足 {args.count}')
    selected = [candidates[i] for i in sorted(random.Random(args.seed).sample(range(len(candidates)), args.count))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v,_ in selected), encoding='utf-8')
    paths = Counter('both' if all(m['msg'] for m in v['msgs']) else 'user_only' if v['msgs'][0]['msg'] else 'assistant_only' for v,_ in selected)
    report = {'source':str(args.source.resolve()), 'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'seed':args.seed, 'unique_candidates':len(candidates), 'selected_count':len(selected),
              'counts':dict(counts), 'paths':dict(paths),
              'note':'llm user_pruning/assistant_pruning；完整输入去重后随机抽样，按源顺序写入。功能测试，可能与训练数据重合。',
              'rows':[dict(output_line=i, **ref) for i,(_,ref) in enumerate(selected,1)]}
    args.output.with_suffix('.sources.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
