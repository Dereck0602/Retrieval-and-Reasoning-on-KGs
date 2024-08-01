model='common_base_model/llama2-7b-chat'
data='instruction_data/instruction_entity_relation_yaml.json'
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
    --output_dir output/instruct_yaml_2e-6 > log/agnoistic.log
