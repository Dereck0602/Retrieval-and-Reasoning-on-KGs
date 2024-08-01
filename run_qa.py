import json
import argparse
import torch
from generate_qa import GenQA
import copy
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import re
import string
import sys
from collections import Counter
import torch.nn as nn
from data_utils import load_data
from vllm import LLM, SamplingParams


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to the pretrained model.")
    parser.add_argument("--k", type=int, default=1, help="number of demonstrations for each test sample.")
    parser.add_argument("--selecting_strategy", type=str,
                        help="method for demonstration selection, should be in ['random', 'TopK'].")
    parser.add_argument("--dataset", type=str, help="which dataset to inference should be in ['CDNLI', 'CLAmazon', "
                                                    "'CommonSense', 'MMLU', 'XNLI'].")
    parser.add_argument("--prompt", type=str, help="prompt type.")
    parser.add_argument("--save_dir", type=str, help="the path to save.")
    parser.add_argument("--use_graph", default=False, action="store_true", help='use graph aiding reasoning or not')
    parser.add_argument("--case_study", default=False, action="store_true", help='case study or not')
    parser.add_argument("--only_generate_cot", default=False, action="store_true", help='only generate cot or not')
    parser.add_argument("--rethink", default=False, action="store_true", help='use the graph to rethink the generated cot or not')
    parser.add_argument("--graph_with_json", default=False, action="store_true", help='use the json format or not')
    parser.add_argument("--graph_with_yaml", default=False, action="store_true", help='use the yaml format or not')
    parser.add_argument("--code_prompt", default=False, action="store_true", help='use the yaml format or not')
    args = parser.parse_args()
    return args

def get_embedding(sentence):
    embedding = sentence_model.encode([sentence], convert_to_tensor=True)
    return embedding


def bank_init(args):
    """Initiallize the demonstration bank and the corresponding embeddings."""
    train_dataset, _ = load_data(args.dataset)
    #demonstration_bank = [i["query"] + i["label"] for i in train_dataset]
    sentences = [i["question"] for i in train_dataset]
    print("Encoding {} train samples".format(len(sentences)))
    embedding_bank = sentence_model.encode(sentences, convert_to_tensor=True, batch_size=1024)
    print("Demonstration bank prepared.")
    return embedding_bank



if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if 'phi' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                device_map='auto', trust_remote_code=True).eval()  #load_in_8bit=True
    
    elif '70b' in args.model:
        model = LLM(model=args.model,tensor_parallel_size=2, seed=0)
    else:
        model = LLM(model=args.model, seed=0)
        
    results = []
    train_data, test_data = load_data(args.dataset)
    
    
    inferencer = GenQA(args=args, tokenizer=tokenizer, model=model, train_dataset=train_data,
                        test_dataset=test_data, device=device, case_study=args.case_study)
    prediction = inferencer.run()
    #print(reference)
