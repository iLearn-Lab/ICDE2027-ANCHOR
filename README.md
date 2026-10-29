# ANCHOR: Category-Anchored Large Language Models for Next POI Recommendation

> **ANCHOR** bridges high-level mobility intent and concrete POI prediction with category anchors, multi-source candidate grounding, and history-aware reranking.

## Authors

**Lexi Bai**<sup>1</sup>, **Xiaopeng Zeng**<sup>1</sup>, **Xun Zhou**<sup>1</sup>, **Yuanshao Zhu**<sup>2,*</sup>, and **James Jianqiao Yu**<sup>1,*</sup>

<sup>1</sup> `Harbin Institute of Technology (Shenzhen)`  
<sup>2</sup> `City University of Hong Kong`  

## Links

- **Paper**: [`paper.pdf`](paper.pdf)

---

## Introduction

This repository is the implementation accompanying the paper **Category-Anchored Large Language Models for Next POI Recommendation**.

Next POI recommendation must select a concrete, localized POI from a sparse and long-tailed entity space. Directly asking a large language model (LLM) to generate POI identifiers can therefore produce invalid or poorly grounded entities even when the model understands the user's visiting intent.

ANCHOR treats category-level mobility intent as a semantic anchor. The complete pipeline is:

1. **Semantic intent pre-estimation**: a LoRA-adapted LLM predicts the target POI category.
2. **Multi-source candidate grounding**: category anchors are combined with historical visits and user preference signals to construct a concrete candidate pool.
3. **History-aware reranking**: a Transformer reranker scores candidates using user history, target context, POI embeddings, and candidate recall features.

This repository provides the training, inference, candidate-generation, and evaluation scripts for the pipeline. It also provides a tiny NYC sample so that the data flow and output formats can be checked without downloading the full datasets or model checkpoints.

## Highlights

- Category-anchored semantic-to-entity grounding for next POI recommendation.
- LoRA fine-tuning of a local instruction LLM for category prediction.
- Three-source candidate recall from LLM categories, history memory, and user preferences.
- LightGCN POI-transition embeddings and a history-aware Transformer reranker.
- End-to-end commands for training, candidate construction, and test evaluation.

## Method / Framework

The method decomposes next POI recommendation into semantic intent estimation, candidate grounding, and entity-level reranking:

```text
Trajectory history
        │
        ▼
LLM category prediction ──► category anchors
                                  │
                                  ▼
                 LLM + history + user-preference recall
                                  │
                                  ▼
                    history-aware Transformer reranker
                                  │
                                  ▼
                         ranked next POIs
```

The complete model diagram and method details are provided in [`paper.pdf`](paper.pdf), especially Fig. 2.


## Installation

### 1. Create the environment

Python 3.10 is recommended. Stage 1 needs a CUDA-capable environment that can load the base LLM with 4-bit quantization.

```bash
conda create -n anchor python=3.10 -y
conda activate anchor
pip install -r requirements.txt
```

### 2. Configure paths

Set a local Hugging Face-compatible instruction model before running Stage 1. The paper experiments use Qwen2.5-7B-Instruct.

```bash
export BASE_MODEL=/path/to/Qwen2.5-7B-Instruct
export ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

Set the sample paths. For full experiments, replace these paths with the corresponding full-dataset splits.

```bash
export DATA_DIR="$ROOT/data/NYC"
export TRAIN_CSV="$DATA_DIR/nyc_train_sample.csv"
export VAL_CSV="$DATA_DIR/nyc_val_sample.csv"
export TEST_CSV="$DATA_DIR/nyc_test_sample.csv"
```

## Checkpoints / Models

No trained checkpoints are included. Stage 1 requires a local base model specified by `BASE_MODEL`; the training command writes the LoRA adapter and classifier head under `output_nyc/cls_head_laststep/best`.

The remaining stages consume the generated POI embeddings, candidate tensors, and reranker checkpoint from the preceding stages. These generated artifacts are ignored by `.gitignore`.

## Dataset / Benchmark

The repository includes only a tiny NYC sample:

```text
data/NYC/nyc_train_sample.csv
data/NYC/nyc_val_sample.csv
data/NYC/nyc_test_sample.csv
```

Each CSV contains the following fields:

| Column | Description |
| --- | --- |
| `user_id` | User identifier |
| `POI_id` | POI identifier |
| `POI_catid`, `POI_catid_code`, `POI_catname` | POI category fields |
| `latitude`, `longitude` | POI coordinates |
| `UTC_time`, `local_time` | Check-in timestamps |
| `day_of_week`, `norm_in_day_time`, `norm_day_shift` | Temporal features |
| `trajectory_id` | Trajectory identifier |
| `norm_relative_time` | Relative time within a trajectory |
| `split_type` | Dataset split label |

The sample is intended for checking data flow and output formats. It is not large enough to reproduce the paper's reported metrics. Full-dataset experiments require the corresponding preprocessed check-in data, using the same schema, and a local base LLM; neither is bundled here.

## Usage

Run all commands from the repository root. The commands below use the bundled sample and exercise the complete pipeline.

### Stage 1: Train the LLM category classifier

```bash
mkdir -p "$ROOT/output_nyc/cls_head_laststep" \
         "$ROOT/output_nyc/top50_laststep"

python train_cls_head_laststep.py \
  --model_dir "$BASE_MODEL" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --out_dir "$ROOT/output_nyc/cls_head_laststep" \
  --history_limit 50 \
  --max_len 2560 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 5e-5 \
  --epochs 5 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --use_4bit \
  --bf16
```

Export top-50 category predictions for the train, validation, and test targets:

```bash
python eval_cls_head_topk_laststep.py \
  --base "$BASE_MODEL" \
  --adapter "$ROOT/output_nyc/cls_head_laststep/best" \
  --label_space_json "$ROOT/output_nyc/cls_head_laststep/label_space.json" \
  --train_csv "$TRAIN_CSV" \
  --target_csv "$TRAIN_CSV" \
  --history_csvs "$TRAIN_CSV" \
  --history_limit 50 \
  --max_len 2560 \
  --batch_size 4 \
  --use_4bit \
  --bf16 \
  --topk_list "1,3,5,10,20,30,40,50" \
  --save_topk_path "$ROOT/output_nyc/top50_laststep/train_top50_categories.jsonl"
```

```bash
python eval_cls_head_topk_laststep.py \
  --base "$BASE_MODEL" \
  --adapter "$ROOT/output_nyc/cls_head_laststep/best" \
  --label_space_json "$ROOT/output_nyc/cls_head_laststep/label_space.json" \
  --train_csv "$TRAIN_CSV" \
  --target_csv "$VAL_CSV" \
  --history_csvs "$TRAIN_CSV,$VAL_CSV" \
  --history_limit 50 \
  --max_len 2560 \
  --batch_size 4 \
  --use_4bit \
  --bf16 \
  --topk_list "1,3,5,10,20,30,40,50" \
  --save_topk_path "$ROOT/output_nyc/top50_laststep/val_top50_categories.jsonl"
```

```bash
python eval_cls_head_topk_laststep.py \
  --base "$BASE_MODEL" \
  --adapter "$ROOT/output_nyc/cls_head_laststep/best" \
  --label_space_json "$ROOT/output_nyc/cls_head_laststep/label_space.json" \
  --train_csv "$TRAIN_CSV" \
  --target_csv "$TEST_CSV" \
  --history_csvs "$TRAIN_CSV,$VAL_CSV,$TEST_CSV" \
  --history_limit 50 \
  --max_len 2560 \
  --batch_size 4 \
  --use_4bit \
  --bf16 \
  --topk_list "1,3,5,10,20,30,40,50" \
  --save_topk_path "$ROOT/output_nyc/top50_laststep/test_top50_categories.jsonl"
```

### Stage 2: Build the POI graph and train LightGCN

```bash
mkdir -p "$ROOT/output_nyc/poi_graph_clean/lightgcn_run"

python build_poi_graph.py \
  --train_csv "$TRAIN_CSV" \
  --out_dir "$ROOT/output_nyc/poi_graph_clean"

python train_poi_lightgcn.py \
  --graph_data "$ROOT/output_nyc/poi_graph_clean/poi_graph_data.pt" \
  --poi_index "$ROOT/output_nyc/poi_graph_clean/poi_index.json" \
  --out_dir "$ROOT/output_nyc/poi_graph_clean/lightgcn_run" \
  --eval_csv "$TRAIN_CSV"
```

Using the training split as `--eval_csv` in this sample run is intentional: it causes the script to save `poi_embeddings_best.pt` for the downstream stages. Use a separate validation split when selecting graph-training checkpoints for a full experiment.

### Stage 3: Build three-source candidate pools

```bash
mkdir -p \
  "$ROOT/output_nyc/rerank_candidates_three_source/train" \
  "$ROOT/output_nyc/rerank_candidates_three_source/val" \
  "$ROOT/output_nyc/rerank_candidates_three_source/test"
```

```bash
python build_three_source_candidates.py \
  --poi_embeddings "$ROOT/output_nyc/poi_graph_clean/lightgcn_run/poi_embeddings_best.pt" \
  --catalog_csv "$TRAIN_CSV" \
  --target_csv "$TRAIN_CSV" \
  --history_csvs "$TRAIN_CSV" \
  --topk_jsonl "$ROOT/output_nyc/top50_laststep/train_top50_categories.jsonl" \
  --out_dir "$ROOT/output_nyc/rerank_candidates_three_source/train" \
  --split_name train \
  --history_limit 50 \
  --llm_topk 10 \
  --candidate_size 2000 \
  --llm_category_quota 800 \
  --history_quota 100 \
  --user_pref_category_quota 200 \
  --user_pref_cat_topm 20 \
  --enabled_sources llm,history,user_pref
```

```bash
python build_three_source_candidates.py \
  --poi_embeddings "$ROOT/output_nyc/poi_graph_clean/lightgcn_run/poi_embeddings_best.pt" \
  --catalog_csv "$TRAIN_CSV" \
  --target_csv "$VAL_CSV" \
  --history_csvs "$TRAIN_CSV,$VAL_CSV" \
  --topk_jsonl "$ROOT/output_nyc/top50_laststep/val_top50_categories.jsonl" \
  --out_dir "$ROOT/output_nyc/rerank_candidates_three_source/val" \
  --split_name val \
  --history_limit 50 \
  --llm_topk 10 \
  --candidate_size 2000 \
  --llm_category_quota 800 \
  --history_quota 100 \
  --user_pref_category_quota 200 \
  --user_pref_cat_topm 20 \
  --enabled_sources llm,history,user_pref
```

```bash
python build_three_source_candidates.py \
  --poi_embeddings "$ROOT/output_nyc/poi_graph_clean/lightgcn_run/poi_embeddings_best.pt" \
  --catalog_csv "$TRAIN_CSV" \
  --target_csv "$TEST_CSV" \
  --history_csvs "$TRAIN_CSV,$VAL_CSV,$TEST_CSV" \
  --topk_jsonl "$ROOT/output_nyc/top50_laststep/test_top50_categories.jsonl" \
  --out_dir "$ROOT/output_nyc/rerank_candidates_three_source/test" \
  --split_name test \
  --history_limit 50 \
  --llm_topk 10 \
  --candidate_size 2000 \
  --llm_category_quota 800 \
  --history_quota 100 \
  --user_pref_category_quota 200 \
  --user_pref_cat_topm 20 \
  --enabled_sources llm,history,user_pref
```

### Stage 4: Train the history-aware reranker

```bash
mkdir -p "$ROOT/output_nyc/reranker_three_source/history_aware"

python train_best_reranker.py \
  --train_candidate_dir "$ROOT/output_nyc/rerank_candidates_three_source/train" \
  --val_candidate_dir "$ROOT/output_nyc/rerank_candidates_three_source/val" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --catalog_csv "$TRAIN_CSV" \
  --poi_embeddings "$ROOT/output_nyc/poi_graph_clean/lightgcn_run/poi_embeddings_best.pt" \
  --out_dir "$ROOT/output_nyc/reranker_three_source/history_aware" \
  --history_limit 50 \
  --history_summary_len 50 \
  --summary_recency_mode inverse \
  --batch_size 64 \
  --num_workers 4 \
  --epochs 60 \
  --early_stop_patience 10 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --dropout 0.15 \
  --weight_decay 0.02 \
  --lr 1e-5 \
  --coarse_loss_lambda 0.0 \
  --best_metric val_hit1_all \
  --enable_candidate_score \
  --enable_candidate_cat_rank \
  --enable_candidate_gcn_rank \
  --enable_history_candidate_features \
  --enable_candidate_history_attention \
  --disable_user_features \
  --disable_same_cat_prev
```

### Stage 5: Evaluate on the test split

```bash
python eval_best_reranker.py \
  --model_ckpt "$ROOT/output_nyc/reranker_three_source/history_aware/best.pt" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_candidate_dir "$ROOT/output_nyc/rerank_candidates_three_source/test" \
  --test_csv "$TEST_CSV" \
  --catalog_csv "$TRAIN_CSV" \
  --poi_embeddings "$ROOT/output_nyc/poi_graph_clean/lightgcn_run/poi_embeddings_best.pt" \
  --batch_size 64 \
  --num_workers 4 \
  --out_json "$ROOT/output_nyc/reranker_three_source/history_aware/test_eval.json" \
  --out_jsonl "$ROOT/output_nyc/reranker_three_source/history_aware/test_cases.jsonl" \
  --out_csv "$ROOT/output_nyc/reranker_three_source/history_aware/test_cases.csv"
```

## Acknowledgement

This implementation uses PyTorch, Hugging Face Transformers, Datasets, PEFT, BitsAndBytes, and the LightGCN propagation idea. We thank the authors and maintainers of these projects for making their software available to the research community.

## License

No standalone `LICENSE` file is currently included. The repository and bundled sample data are provided for research and reproducibility purposes; contact the authors before redistributing the code or data.
