model='output/llama2_entity_rela_graph2text_text2graph_yaml_graphtune_post_qacot_2e-6_sample5_5epoch'

data='/ossfs/workspace/data/webcwq/'
export CUDA_VISIBLE_DEVICES=0
python run_qa.py \
    --model ${model} \
    --k 8 \
    --dataset ${data} \
    --prompt cot \
    --use_graph \
    --graph_with_yaml \
    --selecting_strategy random \
    --save_dir /ossfs/workspace/cwq_instruct_qacot_cot_answer_output.json
