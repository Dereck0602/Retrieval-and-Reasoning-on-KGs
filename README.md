# Retrieval-and-Reasoning-on-KGs

## Step 1: Train the retrieval model
```
bash finetune_retrieve.sh
```

## Step 2: Train the reasoning model
```
model='common_base_model/llama2-7b-chat'
data='instruction_data/instruction_yaml.json
accelerate launch --config_file default_config.yaml run_clm_finetune.py \
    --model_name_or_path ${model} \
    --train_file ${data} \
    --per_device_train_batch_size 2 \
    --cutoff_len 1024 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-6 \
    --num_train_epochs 2 \
    --preprocessing_num_workers 8 \
    --output_dir output/instruct_yaml_2e-6
```

```
model='output/instruct_yaml_2e-6'
data='instruction_data/graph_yaml.json
accelerate launch --config_file default_config.yaml run_clm_finetune.py \
    --model_name_or_path ${model} \
    --train_file ${data} \
    --per_device_train_batch_size 2 \
    --cutoff_len 1024 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 4 \
    --kg_pretrain \
    --learning_rate 2e-6 \
    --num_train_epochs 2 \
    --preprocessing_num_workers 8 \
    --output_dir output/instruct_kgpretrain_yaml_2e-6
```

```
model='output/instruct_kgpretrain_yaml_2e-6'
data='instruction_data/kgqa_reason_yaml.json
accelerate launch --config_file default_config.yaml run_clm_finetune.py \
    --model_name_or_path ${model} \
    --train_file ${data} \
    --per_device_train_batch_size 2 \
    --cutoff_len 1024 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-6 \
    --num_train_epochs 5 \
    --preprocessing_num_workers 8 \
    --output_dir output/instruct_kgpretrain_kgreason_yaml_2e-6
```

## Step 3: Inference
```
bash run.sh
```
