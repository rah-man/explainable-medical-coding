import argparse
import json
import os
import re
from typing import Any, List

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
    if not s:
        return []

    try:
        parsed = json.loads(s)
        return safe_codes(parsed)
    except Exception:
        pass

    return sorted({x.strip() for x in re.split(r"[,;\s]+", s) if x.strip()})


def parse_json_prediction(text: str) -> List[str]:
    text = text.strip()

    try:
        obj = json.loads(text)
        return safe_codes(obj.get("icd_codes", []))
    except Exception:
        return []


def make_prompt(tokenizer, note: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(note=str(note))},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="./icd_data/mimiciv_icd10/test.parquet")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_file", default="./outputs/test_predictions_vllm.jsonl")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="diagnosis_codes")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    df = pd.read_parquet(args.test_file)
    if args.limit is not None:
        df = df.head(args.limit)

    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        structured_outputs={"json": JSON_SCHEMA},
    )

    lora_request = LoRARequest(
        "olmo3_icd_lora",
        1,
        args.adapter_dir,
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(df), args.batch_size), desc="Predicting"):
            batch = df.iloc[start:start + args.batch_size]

            prompts = [
                make_prompt(tokenizer, row[args.text_col])
                for _, row in batch.iterrows()
            ]

            outputs = llm.generate(
                prompts,
                sampling_params,
                lora_request=lora_request,
            )

            for (_, row), output in zip(batch.iterrows(), outputs):
                raw_pred = output.outputs[0].text
                pred_codes = parse_json_prediction(raw_pred)
                gold_codes = safe_codes(row[args.label_col])

                record = {
                    "note_id": row.get("note_id", None),
                    "subject_id": row.get("subject_id", None),
                    "hadm_id": row.get("_id", None),
                    "gold_codes": gold_codes,
                    "pred_codes": pred_codes,
                    "raw_prediction": raw_pred,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()