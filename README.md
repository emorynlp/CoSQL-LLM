# Merging Models

This directory serves as a repository for the code for merging and training CoCodeS, presented in Advancing Conversational Text-to-SQL: Current Landscape and Future Directions with Large Language Models.

This repository contains scripts to prepare datasets, generate GQR summaries, fine-tune, run zero-shot inference, and merge model checkpoints.

### Create and activate Conda env

```bash
conda create -n codes python=3.8.5
conda activate merging
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Data

Download the following datasets and place them under `data/sft_data_collections/`:

- [**Spider**](https://yale-lily.github.io/spider)
- [**SParC**](https://yale-lily.github.io/sparc)
- [**CoSQL**](https://yale-lily.github.io/cosql)
- [**BIRD**](https://bird-bench.github.io/)

Layout:

```
data/
  sft_data_collections/
    spider/
      ...
    cosql/
      ...
    sparc/
      ...
    bird/
      ...
```

---

## Prepare GQR Summaries

```bash
python3 prepare_summary_prompts.py
```

---

## Prepare Datasets for SFT

```bash
python3 prepare_sft_datasets.py
```

---

## Fine-tuning

Follow the instructions on the [**CodeS**](https://github.com/RUCKBReasoning/codes) GitHub to set up accelerate for finetuning, and download the sic_ckpt folder from there as well.  
Example: fine-tuning **BIRD** CodeS model on **GQR CoSQL**:

```bash
accelerate launch train_causal_lm.py   --per_device_train_batch_size 1   --block_size 4096   --seed 42   --pretrained_model_name_or_path seeklhy/codes-7b-bird   --epochs 4   --lr 5e-6   --warmup_ratio 0.05   --checkpointing_steps 100000   --tensorboard_log_dir ./train_logs/codes-7b-spider-cosql-bird   --mode sft   --output_ckpt_dir ./ckpts/codes-7b-bird-cosql-gqr   --text2sql_data_dir ./data/sft_cosql_train_gpt.json   --table_num 6   --column_num 10
```

---

## Inference

### With GQR summaries

```bash
python -u prepare_zero_shot_summaries.py   --llm_path path/to/ckpt   --sic_path ./sic_ckpts/sic_spider   --table_num 6   --column_num 10   --max_tokens 4096   --max_new_tokens 256   --output_path results.txt
```

### On full history

```bash
python -u prepare_zero_shot.py   --llm_path path/to/ckpt   --sic_path ./sic_ckpts/sic_spider   --table_num 6   --column_num 10   --max_tokens 4096   --max_new_tokens 256   --output_path results.txt
```

---

## Merge Models


To Merge models, use the following:
```bash
python3 merge.py   --llm_path1 path/to/ckpt1   --llm_path2 path/to/ckpt2   --save_path path/to/ckpt_merged
```
After this, you can run inference on the new model.
