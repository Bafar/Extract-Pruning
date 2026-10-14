"""Real-weight integration smoke test; writes diagnostics and actual sample output."""
import json
from pathlib import Path
from pipeline import ExtractPruning


def main():
    root = Path(__file__).resolve().parent
    p = ExtractPruning()
    sample = json.loads((root/'input_sample.json').read_text(encoding='utf-8'))
    result, trace = p.predict_with_diagnostics(sample)
    (root/'output_sample.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report = [{'case':'user_recipe', 'output':result, 'diagnostics':trace}]
    cases = [('empty', {'msgs':[]}),
             ('short_user', {'msgs':[{'role':'User','msg':'你好'}]}),
             ('assistant_only', {'msgs':[{'role':'Assistant','msg':'请问您需要修改文章，还是总结文章？'}]}),
             ('both_with_file', {'msgs':[{'role':'User','msg':'请记住这个食谱。<input-file-summary>西红柿炒鸡蛋：3个鸡蛋、2个西红柿，先炒蛋盛出，再炒西红柿，最后混合。</input-file-summary>'},
                                       {'role':'Assistant','msg':'请问您需要修改文章，还是总结文章？'}]}),
             ('assistant_again', {'msgs':[{'role':'Assistant','msg':'请问您需要修改文章，还是总结文章？'}]})]
    model_id = None
    for name, data in cases:
        output, diagnostic = p.predict_with_diagnostics(data)
        if p.llm.model is not None:
            if model_id is None: model_id = id(p.llm.model)
            assert model_id == id(p.llm.model), 'base model was reloaded'
        report.append({'case':name,'output':output,'diagnostics':diagnostic})
        print(name, json.dumps(output,ensure_ascii=False),flush=True)
    assert report[0]['output']['assistant_memory_hint'] is None
    assert all(v is None for v in report[1]['output'].values())
    assert report[4]['output']['should_process_user_msg'] is True
    assert report[3]['output'] == report[5]['output'], 'adapter switching changed deterministic output'
    (root/'smoke_test_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
