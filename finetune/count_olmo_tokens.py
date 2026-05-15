import argparse
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


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

def make_messages(note, codes):
    if codes is None:
        codes = []
    elif hasattr(codes, "tolist"):
        codes = codes.tolist()
    elif not isinstance(codes, (list, tuple, set)):
        codes = [str(codes)]

    target = json.dumps(
        {"icd_codes": sorted([str(c) for c in codes if c is not None])},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(note=note)},
        {"role": "assistant", "content": target},
    ]

def summarise(lengths, name):
    arr = np.array(lengths)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"n:        {len(arr)}")
    print(f"min:      {arr.min():.0f}")
    print(f"p25:      {np.percentile(arr, 25):.0f}")
    print(f"median:   {np.percentile(arr, 50):.0f}")
    print(f"mean:     {arr.mean():.1f}")
    print(f"p75:      {np.percentile(arr, 75):.0f}")
    print(f"p90:      {np.percentile(arr, 90):.0f}")
    print(f"p95:      {np.percentile(arr, 95):.0f}")
    print(f"p99:      {np.percentile(arr, 99):.0f}")
    print(f"max:      {arr.max():.0f}")
    print(f">4096:    {(arr > 4096).sum()} ({(arr > 4096).mean() * 100:.2f}%)")
    print(f">8192:    {(arr > 8192).sum()} ({(arr > 8192).mean() * 100:.2f}%)")
    print(f">16384:   {(arr > 16384).sum()} ({(arr > 16384).mean() * 100:.2f}%)")


def count_file(path, tokenizer, text_col, label_col, batch_size):
    df = pd.read_parquet(path)

    lengths = []

    for start in tqdm(range(0, len(df), batch_size), desc=f"Counting {path}"):
        batch = df.iloc[start:start + batch_size]

        texts = []
        for _, row in batch.iterrows():
            messages = make_messages(row[text_col], row[label_col])
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(formatted)

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )

        lengths.extend(encoded["length"])

    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./icd_data/mimiciv_icd10/train.parquet")
    parser.add_argument("--val_file", default="./icd_data/mimiciv_icd10/val.parquet")
    parser.add_argument("--test_file", default="./icd_data/mimiciv_icd10/test.parquet")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="diagnosis_codes")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_csv", default="./outputs/olmo_token_length_stats.csv")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    all_rows = []

    for name, path in [
        ("train", args.train_file),
        ("val", args.val_file),
        ("test", args.test_file),
    ]:
        lengths = count_file(
            path=path,
            tokenizer=tokenizer,
            text_col=args.text_col,
            label_col=args.label_col,
            batch_size=args.batch_size,
        )

        summarise(lengths, name)

        for length in lengths:
            all_rows.append({"split": name, "num_tokens": length})

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved per-document token lengths to: {args.output_csv}")


if __name__ == "__main__":
    main()