#!/usr/bin/env python3
"""Run offline predictions from JSONL: one sample JSON object per line."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


OPEN = "<input-file-summary>"
CLOSE = "</input-file-summary>"
FULL_TAG = re.compile(r"<input-file-summary>.*?</input-file-summary>", re.DOTALL)
INTENT_WEIGHTS = {"修改": 2, "翻译": 2, "总结": 1.5, "润色": 1.5, "概括": 1.5,
                  "整理": 1, "改写": 1.5, "记住": 1, "看看": 0.5}
BOUNDARY_WORDS = ("下面", "以下", "这段", "这个", "原文", "转发", "复制", "粘贴")
URL = re.compile(r"https?://[^\s<>]+", re.I)
FIRST_PERSON = re.compile(r"我们|我的|我")


def extract_features(msg: str) -> list[float]:
    code_count = msg.count("```") // 2
    lines = msg.splitlines()
    table_count = sum(bool(re.match(r"^\s*\|.+\|\s*$", line)) for line in lines)
    quote_count = sum(bool(re.match(r"^\s*>\s*\S", line)) for line in lines)
    intent = sum(msg.count(word) * weight for word, weight in INTENT_WEIGHTS.items())
    boundary = sum(msg.count(word) for word in BOUNDARY_WORDS)
    first_person = len(FIRST_PERSON.findall(msg)) / max(len(msg), 1)
    paragraphs = len([part for part in re.split(r"\n\s*\n|\n", msg) if part.strip()])
    return [min(code_count, 5) / 5, min(table_count, 3) / 3,
            min(quote_count, 3) / 3, min(len(URL.findall(msg)), 5) / 5,
            min(intent, 10) / 10, min(boundary, 5) / 5, first_person,
            min(paragraphs, 10) / 10, min(len(msg), 3000) / 3000]


def route_message(msg: str | None) -> tuple[str, bool | None, str | None]:
    if msg is None or not msg.strip():
        return "empty", None, None
    opens, closes = msg.count(OPEN), msg.count(CLOSE)
    matches = list(FULL_TAG.finditer(msg))
    if opens != closes or len(matches) != opens or (opens == 0 and "input-file-summary" in msg):
        return "model", None, "malformed_file_summary_tag"
    if matches:
        return "file_summary", True, None
    return "model", None, None


def encode_message(tokenizer, msg: str, max_length: int) -> tuple[dict, bool]:
    if max_length < 8:
        raise ValueError("max_length must be >= 8")
    ids = tokenizer.encode(msg, add_special_tokens=False)
    truncated = len(ids) > max_length - 2
    if truncated:
        capacity = max_length - 2
        head = min(200, capacity // 2)
        tail = min(100, (capacity - head) // 2)
        middle = capacity - head - tail
        inner = ids[head:len(ids) - tail]
        sampled = [inner[min(int((i + .5) * len(inner) / middle), len(inner) - 1)]
                   for i in range(middle)]
        ids = ids[:head] + sampled + ids[-tail:]
    sequence = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    mask = [1] * len(sequence)
    padding = max_length - len(sequence)
    sequence += [tokenizer.pad_token_id] * padding
    mask += [0] * padding
    return {"input_ids": sequence, "attention_mask": mask,
            "token_type_ids": [0] * max_length}, truncated


def load_model(model_path: Path, config: dict, device: str):
    model_dir = model_path.parent
    bert_dir = (model_dir / config["bert_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True, use_fast=True)
    # best.safetensors contains the full fine-tuned encoder plus classifier, so only the
    # architecture config is needed before loading it; no duplicate base weights.
    bert_config = AutoConfig.from_pretrained(bert_dir, local_files_only=True)
    bert = AutoModel.from_config(bert_config)

    class BertFeatureGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = bert
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.bert.config.hidden_size + 9, config["head_size"]),
                torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(config["head_size"], 1))

        def forward(self, input_ids, attention_mask, token_type_ids, features):
            encoded = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            return self.head(torch.cat([encoded.last_hidden_state[:, 0], features], dim=1)).squeeze(-1)

    model = BertFeatureGate().to(device)
    model.load_state_dict(load_file(str(model_path), device=device))
    model.eval()
    return tokenizer, model


def sample_id_sort_key(sample_id):
    if isinstance(sample_id, int) and not isinstance(sample_id, bool):
        return (0, sample_id)
    return (1, sample_id)


def read_jsonl(path: Path) -> list[dict]:
    samples = []
    seen_ids = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            if "sample_id" not in sample:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            sample_id = sample["sample_id"]
            if not ((isinstance(sample_id, int) and not isinstance(sample_id, bool))
                    or isinstance(sample_id, str)):
                raise ValueError(f"{path}:{line_number}: sample_id must be a string or integer")
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
            seen_ids.add(sample_id)
            if "current_conversation" not in sample:
                raise ValueError(f"{path}:{line_number}: missing current_conversation")
            message = sample["current_conversation"]
            if message is not None and not isinstance(message, str):
                raise ValueError(f"{path}:{line_number}: current_conversation must be a string or null")
            samples.append(sample)
    if not samples:
        raise ValueError(f"no samples found in {path}")
    return sorted(samples, key=lambda sample: sample_id_sort_key(sample["sample_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL file; each line has sample_id and current_conversation")
    parser.add_argument("--model", type=Path, required=True,
                        help="SafeTensors weight file; config.json must be in the same directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="output JSONL file")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"model weight file not found: {args.model}")
    config_path = args.model.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")

    samples = read_jsonl(args.input)
    prepared = [(sample["sample_id"], sample["current_conversation"]) for sample in samples]

    results = [None] * len(prepared)
    model_indices = []
    for index, (sample_id, message) in enumerate(prepared):
        route, rule_value, reason = route_message(message)
        if route == "model":
            model_indices.append(index)
        else:
            result = {"sample_id": sample_id, "current_conversation": message,
                      "should_process_user_msg": rule_value,
                      "route": route, "reason": reason}
            results[index] = result

    if model_indices:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = load_model(args.model, config, device)
        batch_size = 16
        for start in range(0, len(model_indices), batch_size):
            indices = model_indices[start:start + batch_size]
            encoded_batch = []
            truncated_flags = []
            feature_batch = []
            for index in indices:
                _, message = prepared[index]
                encoded, truncated = encode_message(tokenizer, message, config["max_length"])
                encoded_batch.append(encoded)
                truncated_flags.append(truncated)
                feature_batch.append(extract_features(message))
            tensors = {
                key: torch.tensor([encoded[key] for encoded in encoded_batch], dtype=torch.long, device=device)
                for key in ("input_ids", "attention_mask", "token_type_ids")
            }
            tensors["features"] = torch.tensor(feature_batch, dtype=torch.float, device=device)
            with torch.no_grad():
                probabilities = torch.sigmoid(model(**tensors)).cpu().tolist()
            for index, probability, truncated in zip(indices, probabilities, truncated_flags):
                sample_id, _ = prepared[index]
                result = {"sample_id": sample_id, "current_conversation": prepared[index][1],
                          "should_process_user_msg": probability >= config["threshold"],
                          "probability": probability, "threshold": config["threshold"],
                          "route": "model", "was_truncated": truncated}
                results[index] = result

    text = "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
