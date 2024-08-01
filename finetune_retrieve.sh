save_path='/ossfs/workspace/model/retrieve_kg_cot_neighbor_model'
model_path='/ossfs/workspace/datacube-nas/model/bge-v1.5-en-base'
train_data='/ossfs/workspace/model/kg_data/retrieve_neighbor_cot_data_all.jsonl'

torchrun --nproc_per_node 8 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ${save_path} \
--model_name_or_path ${model_path} \
--train_data ${train_data} \
--learning_rate 1e-5 \
--bf16 \
--num_train_epochs 5 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \
--deepspeed ds_config.json \
--train_group_size 6 \
--negatives_cross_device \
--logging_steps 200 \
--save_steps 5000 \
--query_instruction_for_retrieval "" 
