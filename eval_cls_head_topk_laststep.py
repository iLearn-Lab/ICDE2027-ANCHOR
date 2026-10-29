import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

from laststep_data_utils import build_cls_examples, build_laststep_samples, concat_history_frames, load_checkin_frame


def safe_set_pad_token(model, pad_token_id: int):
    model.config.pad_token_id = pad_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.pad_token_id = pad_token_id
        except Exception:
            pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--history_csvs", type=str, default="")
    ap.add_argument("--history_limit", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_4bit", action="store_true", default=True)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--topk_list", type=str, default="1,3,5,10,20,30,40,50")
    ap.add_argument("--save_topk_path", type=str, default="")
    ap.add_argument("--label_space_json", type=str, default="")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    train_df = load_checkin_frame(args.train_csv, "train")
    target_source = os.path.splitext(os.path.basename(args.target_csv))[0]
    target_df = load_checkin_frame(args.target_csv, target_source)

    history_frames = []
    target_in_history = False
    if args.history_csvs.strip():
        for idx, path in enumerate([x for x in args.history_csvs.split(",") if x.strip()]):
            norm = os.path.abspath(path.strip())
            if norm == os.path.abspath(args.target_csv):
                history_frames.append(target_df)
                target_in_history = True
            elif norm == os.path.abspath(args.train_csv):
                history_frames.append(train_df)
            else:
                history_frames.append(load_checkin_frame(norm, f"hist{idx}"))
    else:
        history_frames.append(train_df)
    if not target_in_history and os.path.abspath(args.target_csv) != os.path.abspath(args.train_csv):
        history_frames.append(target_df)
    history_df = concat_history_frames(history_frames)

    label_space_json = args.label_space_json
    if not label_space_json:
        adapter_dir = os.path.abspath(args.adapter)
        parent_candidate = os.path.join(os.path.dirname(adapter_dir), "label_space.json")
        if os.path.exists(parent_candidate):
            label_space_json = parent_candidate

    if label_space_json and os.path.exists(label_space_json):
        with open(label_space_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        label2id = {str(k): int(v) for k, v in saved["label2id"].items()}
        id2label = {int(k): str(v) for k, v in saved["id2label"].items()}
    else:
        cats = sorted(train_df["POI_catname"].astype(str).unique().tolist())
        label2id = {c: i for i, c in enumerate(cats)}
        id2label = {i: c for c, i in label2id.items()}

    samples, _ = build_laststep_samples(
        target_df,
        history_df,
        history_limit=args.history_limit,
        require_history=True,
    )
    examples, dropped = build_cls_examples(samples, label2id)
    print(f"Examples: {len(examples)} (dropped: {dropped})")
    ds = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(ex):
        t = tokenizer(ex["text"], truncation=True, max_length=args.max_len)
        t["labels"] = ex["labels"]
        return t

    ds_tok = ds.map(tok_fn, remove_columns=ds.column_names)

    bnb = None
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
        dtype=dtype,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    safe_set_pad_token(model, tokenizer.pad_token_id)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]
    max_k = max(topk_list)
    rows = []
    correct = {k: 0 for k in topk_list}

    for start in range(0, len(ds_tok), args.batch_size):
        end = min(start + args.batch_size, len(ds_tok))
        batch_list = []
        meta_list = [ds[i] for i in range(start, end)]
        for i in range(start, end):
            item = ds_tok[i]
            batch_list.append(
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                    "labels": item["labels"],
                }
            )
        batch = collator(batch_list)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.float().detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        top_idx = np.argsort(-logits, axis=1)[:, :max_k]

        for r in range(len(meta_list)):
            for k in topk_list:
                if labels[r] in top_idx[r, :k]:
                    correct[k] += 1
            rows.append(
                {
                    "trajectory_id": meta_list[r]["trajectory_id"],
                    "target_idx_in_traj": meta_list[r]["target_idx_in_traj"],
                    "target_poi_id": meta_list[r]["target_poi_id"],
                    "true_catname": meta_list[r]["true_catname"],
                    "user_id": meta_list[r]["user_id"],
                    "history_len": meta_list[r]["history_len"],
                    "topk_catnames": [id2label[int(i)] for i in top_idx[r].tolist()],
                }
            )

    metrics = {f"acc@{k}": correct[k] / len(ds_tok) for k in topk_list}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_topk_path:
        with open(args.save_topk_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
