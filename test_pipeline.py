"""Routing and output-contract tests; no weights or GPU required."""
import unittest
from pipeline import ExtractPruning, FIELDS, normalize_input, parse_assistant


class FakeGate:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        return self.value, {'route': 'test'}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, side, payload):
        self.calls.append(side)
        return ('{"assistant_memory_hint":"询问需求。","assistant_memory_type":"question"}'
                if side == 'assistant' else '食谱<|field_sep|>我提供了一份食谱。'), {}


class ContractTests(unittest.TestCase):
    def pipeline(self, gate=True):
        p = ExtractPruning()
        p.gate, p.llm = FakeGate(gate), FakeLLM()
        return p

    def payload(self, user='', assistant=''):
        return {'msgs': [{'role':'User','msg':user}, {'role':'Assistant','msg':assistant}]}

    def test_empty(self):
        p = self.pipeline()
        self.assertEqual(p.predict(self.payload(None, '  ')), dict.fromkeys(FIELDS))
        self.assertEqual(p.gate.calls, 0)
        self.assertEqual(p.llm.calls, [])

    def test_gate_false(self):
        p = self.pipeline(False)
        r = p.predict(self.payload('你好'))
        self.assertIs(r['should_process_user_msg'], False)
        self.assertIsNone(r['processed_user_msg'])
        self.assertEqual(p.llm.calls, [])

    def test_assistant_only(self):
        p = self.pipeline()
        r = p.predict(self.payload('', '有什么需要帮助？'))
        self.assertIsNone(r['should_process_user_msg'])
        self.assertEqual(r['assistant_memory_type'], 'question')
        self.assertEqual(p.gate.calls, 0)

    def test_both(self):
        p = self.pipeline()
        r = p.predict(self.payload('食谱', '有什么需要帮助？'))
        self.assertEqual(set(r), set(FIELDS))
        self.assertEqual(p.llm.calls, ['assistant', 'user'])
        self.assertTrue(all(v is not None for v in r.values()))

    def test_false_still_assistant(self):
        p = self.pipeline(False)
        r = p.predict(self.payload('你好', '有什么需要帮助？'))
        self.assertEqual(p.llm.calls, ['assistant'])
        self.assertIsNone(r['processed_user_topic_entity_hint'])

    def test_validation(self):
        for payload in [{'msgs':[{'role':'User','msg':7}]},
                        {'msgs':[{'role':'system','msg':'x'}]},
                        {'msgs':[{'role':'User','msg':'x'}]*2}]:
            with self.assertRaises(ValueError): normalize_input(payload)
        with self.assertRaises(ValueError): parse_assistant('{}')

    def test_no_null_on_failure(self):
        p = self.pipeline()
        p.llm.generate = lambda *a: ('bad output', {})
        with self.assertRaises(ValueError): p.predict(self.payload('食谱'))


if __name__ == '__main__':
    unittest.main()
