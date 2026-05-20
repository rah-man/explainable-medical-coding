import argparse
import json
import os
import re
from collections import Counter
from typing import Any, List, Dict, Set

import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


BASE_MODEL_ID = "allenai/Olmo-3-7B-Instruct"

SYSTEM_PROMPT = """You are a clinical coding assistant.
Your task is to assign ICD diagnosis codes from a discharge summary.

Return JSON only.
Do not explain.
Do not include any text outside the JSON object.

The JSON schema is:
{"icd_codes": ["CODE1", "CODE2"]}
"""

USER_TEMPLATE = """Assign ICD codes to the following discharge summary.

Discharge summary:
{note}
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "icd_codes": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["icd_codes"],
    "additionalProperties": False,
}


def safe_codes(value: Any) -> List[str]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return sorted({str(x).strip() for x in value if x is not None and str(x).strip()})

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = json.loads(s)
        return safe_codes(parsed)
    except Exception:
        pass

    return sorted({x.strip() for x in re.split(r"[,;\s]+", s) if x.strip()})


def parse_json_prediction(text: str) -> tuple[List[str], bool]:
    text = text.strip()
    try:
        obj = json.loads(text)
        return safe_codes(obj.get("icd_codes", [])), True
    except Exception:
        return [], False


def fixed_prompt_token_length(tokenizer) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(note="")},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def truncate_note_for_generation(tokenizer, note: str, note_budget: int) -> str:
    note_ids = tokenizer(
        str(note),
        add_special_tokens=False,
        truncation=True,
        max_length=note_budget,
    )["input_ids"]
    return tokenizer.decode(note_ids, skip_special_tokens=False)


def make_prompt(tokenizer, note: str, note_budget: int) -> str:
    truncated_note = truncate_note_for_generation(tokenizer, note, note_budget)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(note=truncated_note)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def update_metrics(counts: Dict[str, Any], gold: Set[str], pred: Set[str], valid_json: bool) -> None:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn
    counts["exact_match"] += int(gold == pred)
    counts["n"] += 1
    counts["invalid_json"] += int(not valid_json)
    counts["empty_pred"] += int(len(pred) == 0)
    counts["gold_code_count"] += len(gold)
    counts["pred_code_count"] += len(pred)

    for code in gold | pred:
        counts["per_code"][code]["tp"] += int(code in gold and code in pred)
        counts["per_code"][code]["fp"] += int(code not in gold and code in pred)
        counts["per_code"][code]["fn"] += int(code in gold and code not in pred)


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def finalise_metrics(counts: Dict[str, Any]) -> Dict[str, Any]:
    micro = prf(counts["tp"], counts["fp"], counts["fn"])

    per_code_scores = []
    for code, c in counts["per_code"].items():
        score = prf(c["tp"], c["fp"], c["fn"])
        per_code_scores.append(score)

    macro = {
        "precision": sum(s["precision"] for s in per_code_scores) / len(per_code_scores) if per_code_scores else 0.0,
        "recall": sum(s["recall"] for s in per_code_scores) / len(per_code_scores) if per_code_scores else 0.0,
        "f1": sum(s["f1"] for s in per_code_scores) / len(per_code_scores) if per_code_scores else 0.0,
    }

    n = counts["n"]
    return {
        "n_examples": n,
        "micro": micro,
        "macro": macro,
        "exact_match_rate": counts["exact_match"] / n if n else 0.0,
        "invalid_json_rate": counts["invalid_json"] / n if n else 0.0,
        "empty_prediction_rate": counts["empty_pred"] / n if n else 0.0,
        "mean_gold_codes": counts["gold_code_count"] / n if n else 0.0,
        "mean_pred_codes": counts["pred_code_count"] / n if n else 0.0,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "n_codes_macro_averaged": len(per_code_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="./icd_data/mimiciv_icd10/test.parquet")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_file", default="./outputs/test_predictions_vllm.jsonl")
    parser.add_argument("--metrics_file", default=None)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="diagnosis_codes")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if args.metrics_file is None:
        args.metrics_file = os.path.splitext(args.output_file)[0] + "_metrics.json"
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    overhead = fixed_prompt_token_length(tokenizer)
    note_budget = args.max_model_len - args.max_tokens - overhead
    if note_budget <= 0:
        raise ValueError(
            f"max_model_len={args.max_model_len} is too small after reserving "
            f"max_tokens={args.max_tokens} and prompt overhead={overhead}."
        )
    print(f"Prompt overhead tokens: {overhead}")
    print(f"Note token budget: {note_budget}")

    df = pd.read_parquet(args.test_file)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} examples from {args.test_file}")

    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        structured_outputs={"json": JSON_SCHEMA},
    )

    lora_request = LoRARequest("olmo3_icd_lora", 1, args.adapter_dir)

    counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "exact_match": 0,
        "n": 0,
        "invalid_json": 0,
        "empty_pred": 0,
        "gold_code_count": 0,
        "pred_code_count": 0,
        "per_code": Counter(),
    }
    # Counter values need nested dicts, not integers.
    counts["per_code"] = {}

    with open(args.output_file, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(df), args.batch_size), desc="Predicting"):
            batch = df.iloc[start : start + args.batch_size]
            prompts = [make_prompt(tokenizer, row[args.text_col], note_budget) for _, row in batch.iterrows()]
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

            for (_, row), output in zip(batch.iterrows(), outputs):
                raw_pred = output.outputs[0].text
                pred_codes, valid_json = parse_json_prediction(raw_pred)
                gold_codes = safe_codes(row[args.label_col])

                gold_set = set(gold_codes)
                pred_set = set(pred_codes)

                for code in gold_set | pred_set:
                    counts["per_code"].setdefault(code, {"tp": 0, "fp": 0, "fn": 0})
                update_metrics(counts, gold_set, pred_set, valid_json)

                record = {
                    "note_id": row.get("note_id", None),
                    "subject_id": row.get("subject_id", None),
                    "hadm_id": row.get("hadm_id", row.get("_id", None)),
                    "gold_codes": gold_codes,
                    "pred_codes": pred_codes,
                    "valid_json": valid_json,
                    "raw_prediction": raw_pred,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    metrics = finalise_metrics(counts)
    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {args.output_file}")
    print(f"Saved metrics to: {args.metrics_file}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
