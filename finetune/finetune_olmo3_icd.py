import argparse
import json
import os
import re
from typing import Any, List

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_ID = "allenai/Olmo-3-7B-Instruct"

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


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise ValueError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns are: {cols}"
    )


def normalise_codes(value: Any) -> List[str]:
    """
    Convert the repo's label/code format into a list of strings.
    Handles list, tuple, set, JSON string, comma-separated string.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return sorted({str(x).strip() for x in value if str(x).strip()})

    if hasattr(value, "tolist"):
        return normalise_codes(value.tolist())

    s = str(value).strip()

    # Try JSON/list-like strings first.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return normalise_codes(parsed)
    except Exception:
        pass

    # If everything else fails: split on comma / semicolon / whitespace.
    parts = re.split(r"[,;\s]+", s)
    return sorted({p.strip() for p in parts if p.strip()})

def truncate_note_for_budget(
    tokenizer,
    note: str,
    max_seq_length: int,
    max_completion_tokens: int,
) -> str:
    # Build prompt with an empty note to estimate fixed overhead.
    empty_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(note="")},
    ]

    empty_prompt = tokenizer.apply_chat_template(
        empty_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    empty_prompt_ids = tokenizer(
        empty_prompt,
        add_special_tokens=False,
    )["input_ids"]

    note_budget = max_seq_length - max_completion_tokens - len(empty_prompt_ids)

    if note_budget <= 0:
        raise ValueError(
            f"max_seq_length={max_seq_length} is too small after reserving "
            f"{max_completion_tokens} completion tokens."
        )

    note_ids = tokenizer(
        str(note),
        add_special_tokens=False,
        truncation=True,
        max_length=note_budget,
    )["input_ids"]

    return tokenizer.decode(note_ids, skip_special_tokens=False)

def make_prompt_completion(
    example: dict,
    text_col: str,
    label_col: str,
    tokenizer,
    max_seq_length=8192,
    max_completion_tokens=512,
) -> dict:
    """
    Create a prompt/completion example.

    prompt:
      system + user discharge summary + assistant generation prefix

    completion:
      assistant ICD JSON + EOS token

    With SFTConfig(completion_only_loss=True), TRL computes loss on the
    completion only. This avoids training the model to predict the long note.
    """
    note = truncate_note_for_budget(
        tokenizer=tokenizer,
        note=str(example[text_col]).strip(),
        max_seq_length=max_seq_length,
        max_completion_tokens=max_completion_tokens,
    )

    codes = normalise_codes(example[label_col])

    target = json.dumps(
        {"icd_codes": codes},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(note=note)},
    ]

    # OLMo-3 uses a ChatML-like template. add_generation_prompt=True appends
    # the assistant header so that the completion is exactly the assistant text.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    eos = tokenizer.eos_token or "<|endoftext|>"

    return {
        "prompt": prompt,
        "completion": target + eos,
    }


def load_dataframe_and_columns(
    path: str,
    text_col: str | None,
    label_col: str | None,
) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_parquet(path)

    if text_col is None:
        text_col = find_column(
            df,
            candidates=[
                "text",
                "note",
                "notes",
                "discharge_summary",
                "discharge",
                "clinical_note",
            ],
        )

    if label_col is None:
        label_col = find_column(
            df,
            candidates=[
                "labels",
                "codes",
                "icd_codes",
                "diagnosis_codes",
                "target",
                "targets",
                "all_codes",
            ],
        )

    print(f"Loaded {path}")
    print(f"Using text column:  {text_col}")
    print(f"Using label column: {label_col}")
    print(f"Rows: {len(df)}")

    # Drop rows without text or labels.
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    return df, text_col, label_col


def load_and_prepare(
    path: str,
    text_col: str | None,
    label_col: str | None,
    tokenizer,
) -> Dataset:
    df, text_col, label_col = load_dataframe_and_columns(path, text_col, label_col)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.map(
        lambda x: make_prompt_completion(
            x,
            text_col=text_col,
            label_col=label_col,
            tokenizer=tokenizer,
        ),
        remove_columns=ds.column_names,
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./icd_data/mimiciv_icd10/train.parquet")
    parser.add_argument("--val_file", default="./icd_data/mimiciv_icd10/val.parquet")
    parser.add_argument("--output_dir", default="./icd_models/olmo3-7b-instruct-mimiciv-icd10-lora")
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="epoch",
        help="Use 'no' for fastest experiments, 'epoch' for one validation-loss pass per epoch, or 'steps' for periodic validation.",
    )
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--use_4bit", action="store_true")  # should we use QLoRA?
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset creation now happens after loading the tokenizer, because we use
    # tokenizer.apply_chat_template(...) to construct the OLMo-formatted prompt.
    train_ds = load_and_prepare(args.train_file, args.text_col, args.label_col, tokenizer)
    val_ds = None
    if args.eval_strategy != "no":
        val_ds = load_and_prepare(args.val_file, args.text_col, args.label_col, tokenizer)

    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    if args.max_val_samples is not None and val_ds is not None:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))        

    quantization_config = None
    torch_dtype = torch.bfloat16

    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        completion_only_loss=True,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        report_to="none",
        # write checkpoints regularly to a mounted persistent volume.
        # save_safetensors=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
