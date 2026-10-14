"""user_lora 公共模块：配置加载、数据契约校验、文本清洗、分隔符标签编解码、
超长截断、prompt 拼装与 token 化。训练与评估共用同一实现，杜绝两侧漂移。

数据契约：data/user_data_{train,val,test}.jsonl 每行仅含 input/output 两键；
input.msgs = [User, Assistant]（Assistant 允许为空），output 仅含
processed_user_msg / processed_user_topic_entity_hint。
分组与泄漏校验已由 ../split_user_data.py 完成，本模块不依赖 metadata。
"""

import json
import re
from pathlib import Path

# 输出分隔符，必须与 user_prompt.txt 中的字面量一致。
SEP = "<|field_sep|>"
# 超长输入中间截断的占位标记，必须与 user_prompt.txt 中说明一致。
MIDDLE_MARKER = "…【中间内容已截断】…"
# 助手尾注前缀，与 user_prompt.txt 中"关于可选的助手消息尾注"说明一致。
ASSISTANT_NOTE_HEAD = "[助手消息节选] "

FIELDS = ("processed_user_msg", "processed_user_topic_entity_hint")

# 单趟正则去字面转义：逐个反斜杠消费一次，避免顺序替换导致的二次处理。
_UNESCAPE_RE = re.compile(r'\\(["nt\\])')
_UNESCAPE_MAP = {'"': '"', 'n': '\n', 't': ' ', '\\': '\\'}


def load_config(path):
    """加载 JSON 配置，把相对路径按配置文件所在目录展开为绝对路径，并做基础校验。"""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_path", "prompt_file", "output_dir"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg["data"] = {k: str((path.parent / v).resolve()) for k, v in cfg["data"].items()}
    cfg["config_path"] = str(path)
    if cfg["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("dtype 只支持 bfloat16/float16/float32")
    if cfg["max_seq_length"] <= 0:
        raise ValueError("max_seq_length 必须为正数")
    ui = cfg["user_input"]
    if ui["budget"] <= 0 or ui["assistant_note_budget"] <= 0:
        raise ValueError("输入预算与尾注预算必须为正数")
    if not 0 < ui["head_ratio"] < 1:
        raise ValueError("head_ratio 必须在 (0, 1) 之间")
    if ui["use_assistant_note"] and ui["user_budget_with_note"] >= ui["budget"]:
        raise ValueError("开启尾注时 user_budget_with_note 应小于完整 budget")
    if cfg["training"]["save_steps"] != cfg["training"]["eval_steps"]:
        raise ValueError("save_steps 与 eval_steps 必须相等，以便及时保存并加载最佳模型")
    if cfg["lora"]["r"] <= 0:
        raise ValueError("LoRA r 必须为正数")
    return cfg


def read_rows(path):
    """读取并与数据契约逐条校验：仅 input/output 两键，User 非空，输出两字段齐全。"""
    path = Path(path)
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise ValueError(f"空数据集：{path}")
    rows = []
    for line in lines:
        row = json.loads(line)
        if set(row) != {"input", "output"}:
            raise ValueError("数据每行必须仅含 input/output 两键")
        user, assistant = parse_pair(row["input"])
        if not user:
            raise ValueError("训练数据不应包含空 User 消息")
        out = row["output"]
        if not isinstance(out, dict) or set(out) != set(FIELDS):
            raise ValueError("输出字段必须恰为 processed_user_msg / processed_user_topic_entity_hint")
        if not isinstance(out[FIELDS[0]], str) or not out[FIELDS[0]].strip():
            raise ValueError("processed_user_msg 必须为非空字符串")
        if not isinstance(out[FIELDS[1]], str) or not out[FIELDS[1]].strip():
            raise ValueError("gate=True 数据的 processed_user_topic_entity_hint 必须为非空字符串")
        if any(SEP in out[key] for key in FIELDS):
            raise ValueError("目标标签含输出分隔符，需先人工处理冲突")
        rows.append(row)
    return rows


def parse_pair(value):
    """解析 input.msgs 为 (user_text, assistant_text)。"""
    if not isinstance(value, dict) or not isinstance(value.get("msgs"), list):
        raise ValueError("input 必须是含 msgs 列表的对象")
    if len(value['msgs']) != 2 or any(not isinstance(m, dict) for m in value['msgs']) or [m.get('role') for m in value['msgs']] != ['User', 'Assistant']:
        raise ValueError('input.msgs 必须依次且仅含 User、Assistant 两个槽')
    user = assistant = ""
    for message in value["msgs"]:
        role = (message.get("role") or "").lower()
        msg = message.get("msg")
        if msg is not None and not isinstance(msg, str):
            raise ValueError("消息文本必须为字符串或 null")
        text = msg.strip() if isinstance(msg, str) else ""
        if role == "user":
            user = text
        elif role == "assistant":
            assistant = text
    return user, assistant


def user_text(row):
    return parse_pair(row["input"])[0]


def clean_text(text):
    """清洗用户/助手文本：去字面转义、压缩行内空白、删行首尾空白、折叠过多换行。

    此操作有损：会改变代码缩进、字面转义或路径。沿用当前基线设计，代码类需单独验收。
    不删代码围栏或 <input-file-summary> 标签。
    """
    if not text:
        return text
    text = _UNESCAPE_RE.sub(lambda m: _UNESCAPE_MAP[m.group(1)], text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_label(hint, msg):
    """训练标签：锚点在前、整段话在后，用分隔符连接。"""
    return f"{hint.strip()}{SEP}{msg.strip()}"


def parse_output(text):
    """把模型原始输出解析为两字段 dict；失败抛 ValueError 由调用方兜底。"""
    if not isinstance(text, str):
        raise ValueError("输出不是字符串")
    text = text.strip()
    if text.count(SEP) != 1:
        raise ValueError("必须恰有一个分隔符")
    hint, msg = text.split(SEP, 1)
    hint, msg = hint.strip(), msg.strip()
    if not hint or not msg:
        raise ValueError("分隔符两侧为空")
    return {FIELDS[1]: hint, FIELDS[0]: msg}


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def truncate_middle(tokenizer, text, budget, head_ratio):
    """对文本按 token 保头保尾、从中间截断；不足预算原样返回。

    返回字符串。截断后的字符串会被 chat 模板再次 token 化，因 BPE 边界可能产生个位数 token 偏差，
    仅影响占比 1.5% 的超长样本，可接受。
    """
    ids = _encode(tokenizer, text)
    if len(ids) <= budget:
        return text
    marker_ids = _encode(tokenizer, MIDDLE_MARKER)
    available = budget - len(marker_ids)
    if available <= 0:
        return tokenizer.decode(ids[:budget], skip_special_tokens=False)
    head_n = int(available * head_ratio)
    tail_n = available - head_n
    new_ids = ids[:head_n] + marker_ids + ids[-tail_n:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def truncate_tail(tokenizer, text, budget):
    """对文本按 token 只保留尾部预算长度（用于助手尾注）。"""
    ids = _encode(tokenizer, text)
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[-budget:], skip_special_tokens=False)


def build_user_text(row, cfg, tokenizer, user_budget=None):
    """构造送入 user 轮的最终文本（已清洗、已截断、可按开关附尾注）。

    user_budget 可选：仅供显式调试覆盖预算。训练不会根据真实标签长度自动收窄输入。
    """
    ui = cfg["user_input"]
    user = clean_text(user_text(row))
    if not ui["use_assistant_note"]:
        budget = ui["budget"] if user_budget is None else user_budget
        return truncate_middle(tokenizer, user, budget, ui["head_ratio"])
    _, assistant = parse_pair(row["input"])
    budget = ui["user_budget_with_note"] if user_budget is None else user_budget
    user = truncate_middle(tokenizer, user, budget, ui["head_ratio"])
    if assistant:
        note = truncate_tail(tokenizer, clean_text(assistant), ui["assistant_note_budget"])
        return f"{user}\n{ASSISTANT_NOTE_HEAD}{note}"
    return user


def messages_for(prompt, user_text):
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user_text}]


def prompt_ids(tokenizer, prompt, user_text):
    return tokenizer.apply_chat_template(messages_for(prompt, user_text), tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False, return_dict=False)


def _encode_full(tokenizer, prompt, user_line, target):
    """编码完整训练序列（system + user + assistant 目标段），不含 generation prompt。"""
    return tokenizer.apply_chat_template(
        messages_for(prompt, user_line) + [{"role": "assistant", "content": target}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)


def encode_sample(tokenizer, cfg, prompt, row):
    """把单条样本编码为训练序列，返回 {input_ids, attention_mask, labels}。

    labels 只监督分隔符目标段（system+user 输入段与右侧 pad 均为 -100）。
    完整序列超过 max_seq_length 时报错，不使用真实标签长度反向改变 user 输入，
    避免训练与推理前缀不一致。
    """
    hint = row["output"][FIELDS[1]]
    msg = row["output"][FIELDS[0]]
    target = build_label(hint, msg)
    max_len = cfg["max_seq_length"]

    user_line = build_user_text(row, cfg, tokenizer)
    full = _encode_full(tokenizer, prompt, user_line, target)

    if len(full) > max_len:
        raise ValueError(f'完整序列 {len(full)} > max_seq_length {max_len}；请显式调整统一长度预算')

    prefix = prompt_ids(tokenizer, prompt, user_line)
    if full[:len(prefix)] != prefix:
        raise ValueError("chat 模板的生成前缀与训练前缀不一致，禁止错误设置 loss mask")
    labels = [-100] * len(prefix) + full[len(prefix):]
    if not any(x != -100 for x in labels):
        raise ValueError("目标没有可训练 token")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
