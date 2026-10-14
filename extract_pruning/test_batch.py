"""CPU-only batch order/failure/resume tests using a fake inference engine."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import batch_predict
from pipeline import FIELDS


class BatchTests(unittest.TestCase):
    def test_fail_resume_order(self):
        calls = []
        class FakeEngine:
            fail = True
            def __init__(self, config): pass
            def predict_with_diagnostics(self, row):
                text = row['msgs'][0]['msg']
                calls.append(text)
                if text == 'two' and self.fail: raise RuntimeError('synthetic failure')
                result = dict.fromkeys(FIELDS)
                result.update(should_process_user_msg=True, processed_user_msg=text,
                              processed_user_topic_entity_hint=text)
                return result, {}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source, target, config = root/'input.jsonl',root/'output.jsonl',root/'config.json'
            config.write_text('{}',encoding='utf-8')
            source.write_text(''.join(json.dumps({'msgs':[{'role':'User','msg':x}]})+'\n' for x in ('one','two','three')),encoding='utf-8')
            argv=['batch_predict.py','--input',str(source),'--output',str(target),'--config',str(config)]
            with patch.object(batch_predict,'ExtractPruning',FakeEngine), patch.object(batch_predict,'fingerprint',return_value={'test':'fixed'}):
                with patch.object(sys,'argv',argv):
                    with self.assertRaises(RuntimeError): batch_predict.main()
                self.assertEqual(len(target.read_text().splitlines()),1)
                self.assertEqual(json.loads(target.with_suffix('.run.json').read_text())['failed_line'],2)
                FakeEngine.fail=False
                with patch.object(sys,'argv',argv+['--resume']): batch_predict.main()
                values=[json.loads(s) for s in target.read_text().splitlines()]
                self.assertEqual([r['processed_user_msg'] for r in values],['one','two','three'])
                self.assertTrue(all(set(r)==set(FIELDS) for r in values))
                self.assertEqual(calls,['one','two','two','three'])
                self.assertEqual(json.loads(target.with_suffix('.run.json').read_text())['status'],'complete')
                with patch.object(sys,'argv',argv):
                    with self.assertRaises(FileExistsError): batch_predict.main()
                source.write_text(source.read_text()+'{}\n')
                with patch.object(sys,'argv',argv+['--resume']):
                    with self.assertRaises(ValueError): batch_predict.main()


if __name__ == '__main__': unittest.main()
