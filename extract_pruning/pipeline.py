"""Offline five-field inference: BERT gate + one Qwen base with two LoRA adapters."""
from __future__ import annotations

import json
import threading
from pathlib import Path

FIELDS = ('assistant_memory_hint', 'assistant_memory_type', 'should_process_user_msg',
          'processed_user_msg', 'processed_user_topic_entity_hint')
TYPES = {'comfort', 'suggestion', 'recommendation', 'warning', 'instruction',
         'question', 'agreement', 'repetition', 'other'}


def normalize_input(value):
    if not isinstance(value, dict) or set(value) != {'msgs'} or not isinstance(value['msgs'], list):
        raise ValueError('输入必须为仅含 msgs 数组的 JSON 对象')
    roles = {}
    for m in value['msgs']:
        if not isinstance(m, dict) or set(m) != {'role', 'msg'}:
            raise ValueError('每条消息必须包含且仅包含 role、msg')
        role = m['role']
        if not isinstance(role, str) or role.lower() not in ('user', 'assistant'):
            raise ValueError('role 只支持 User/Assistant')
        role = role.lower()
        if role in roles:
            raise ValueError('同一请求每种角色最多一条消息，不猜测会话配对')
        text = m['msg']
        if text is not None and not isinstance(text, str):
            raise ValueError('msg 必须为字符串或 null')
        roles[role] = text if text and text.strip() else ''
    return {'msgs': [{'role': 'User', 'msg': roles.get('user', '')},
                     {'role': 'Assistant', 'msg': roles.get('assistant', '')}]}


def parse_assistant(text):
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != set(FIELDS[:2]):
        raise ValueError('Assistant 输出必须恰含 hint/type 两字段')
    if result['assistant_memory_type'] not in TYPES:
        raise ValueError('Assistant 输出不是合法九分类')
    if not isinstance(result['assistant_memory_hint'], str) or not result['assistant_memory_hint'].strip():
        raise ValueError('非空 Assistant 必须生成非空摘要')
    return result


class Gate:
    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, root
        self.model = self.tokenizer = None

    def predict(self, text):
        import torch
        from gate_runtime import route_message, encode_message, extract_features, load_model
        route, value, reason = route_message(text)
        if route != 'model':
            return value, {'route': route, 'reason': reason}
        device = self.cfg['device']
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.model is None:
            weight = self.root / self.cfg['weights']
            architecture = json.loads(weight.with_name('config.json').read_text(encoding='utf-8'))
            self.tokenizer, self.model = load_model(weight, architecture, device)
        encoded, truncated = encode_message(self.tokenizer, text, self.cfg['max_length'])
        batch = {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in encoded.items()}
        batch['features'] = torch.tensor([extract_features(text)], dtype=torch.float, device=device)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(**batch)).item()
        return probability >= self.cfg['threshold'], {
            'route': route, 'reason': reason, 'probability': probability,
            'threshold': self.cfg['threshold'], 'was_truncated': truncated}


class SharedLLM:
    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, root
        self.model = self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        device = self.cfg['runtime']['device']
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = self.cfg['runtime']['dtype']
        if device == 'cpu':
            dtype = 'float32'
        elif device.startswith('cuda') and dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('当前显卡不支持 BF16；请在 config.json 中选择 float16')
        base_path = self.root / self.cfg['base_model']
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_path, local_files_only=True, dtype=getattr(torch, dtype),
            attn_implementation=self.cfg['runtime']['attention_implementation'])
        model = PeftModel.from_pretrained(base, self.root/self.cfg['assistant']['adapter'],
                                         adapter_name='assistant', is_trainable=False, local_files_only=True)
        model.load_adapter(self.root/self.cfg['user']['adapter'], adapter_name='user',
                           is_trainable=False, local_files_only=True)
        self.model = model.to(device).eval()
        self.device = device

    def generate(self, side, payload):
        import torch
        from user_runtime import build_user_text, clean_text, user_text
        self.load()
        cfg = self.cfg[side]
        prompt = (self.root/cfg['prompt']).read_text(encoding='utf-8')
        trace = {'adapter': side}
        if side == 'user':
            row = {'input': payload}
            content = build_user_text(row, {'user_input': cfg['input']}, self.tokenizer)
            length = len(self.tokenizer.encode(clean_text(user_text(row)), add_special_tokens=False))
            budget = cfg['input']['user_budget_with_note'] if cfg['input']['use_assistant_note'] else cfg['input']['budget']
            trace.update(user_tokens=length, was_truncated=length > budget)
        else:
            content = json.dumps(payload, ensure_ascii=False)
        ids = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if len(ids) + cfg['max_new_tokens'] > cfg['max_seq_length']:
            raise ValueError(f'{side} 输入+生成预算超出 {cfg["max_seq_length"]} token；未静默删尾')
        self.model.set_adapter(side)
        self.model.requires_grad_(False)
        self.model.eval()
        batch = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=batch, attention_mask=torch.ones_like(batch),
                max_new_tokens=cfg['max_new_tokens'], do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True)
        generated = output[0, len(ids):]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        trace.update(input_tokens=len(ids), output_tokens=len(generated))
        if len(generated) >= cfg['max_new_tokens']:
            raise RuntimeError(f'{side} 达到生成长度上限，请检查输出预算；不返回可能不完整的结果')
        return text, trace


class ExtractPruning:
    """Reuse one instance across requests. Lock protects shared adapter switching."""
    def __init__(self, config_path=None):
        config_path = Path(config_path or Path(__file__).with_name('config.json')).resolve()
        self.root = config_path.parent
        self.cfg = json.loads(config_path.read_text(encoding='utf-8'))
        if not 0 <= self.cfg['gate']['threshold'] <= 1:
            raise ValueError('gate.threshold 必须在 [0,1]')
        if not 8 <= self.cfg['gate']['max_length'] <= 512:
            raise ValueError('BERT max_length 必须在 [8,512]')
        for side in ('assistant', 'user'):
            if not 0 < self.cfg[side]['max_new_tokens'] < self.cfg[side]['max_seq_length']:
                raise ValueError('生成预算必须大于0且小于总长度')
        ui = self.cfg['user']['input']
        if not 0 < ui['head_ratio'] < 1 or min(ui['budget'], ui['assistant_note_budget'], ui['user_budget_with_note']) <= 0:
            raise ValueError('User token预算必须为正，head_ratio 必须在 (0,1)')
        if type(ui['use_assistant_note']) is not bool:
            raise ValueError('use_assistant_note 必须为 boolean')
        self.gate = Gate(self.cfg['gate'], self.root)
        self.llm = SharedLLM(self.cfg, self.root)
        self._lock = threading.Lock()

    def predict(self, payload):
        return self.predict_with_diagnostics(payload)[0]

    def predict_with_diagnostics(self, payload):
        from user_runtime import parse_output
        payload = normalize_input(payload)
        result = dict.fromkeys(FIELDS)
        diagnostic = {}
        user, assistant = (m['msg'] for m in payload['msgs'])
        with self._lock:
            if user:
                passed, diagnostic['gate'] = self.gate.predict(user)
                if type(passed) is not bool:
                    raise ValueError('非空 User 的 gate 必须是 boolean')
                result['should_process_user_msg'] = passed
            if assistant:
                raw, diagnostic['assistant'] = self.llm.generate('assistant', payload)
                result.update(parse_assistant(raw))
            if result['should_process_user_msg'] is True:
                raw, diagnostic['user'] = self.llm.generate('user', payload)
                result.update(parse_output(raw))
        return result, diagnostic
