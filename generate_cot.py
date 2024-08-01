import torch
import random
import re
import json
import yaml
from ast import literal_eval
from vllm import LLM, SamplingParams
#from data_utils import load_data
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_data(dataset):
    base_url = dataset
    if 'webqsp' in base_url:
        dataset = load_dataset("parquet", data_files={'train': [base_url+'train-1.parquet', base_url+'train-2.parquet'], 'validation':[base_url+'validation.parquet'], 'test': [base_url+'test-1.parquet', base_url+'test-2.parquet']})
    if 'webcwq' in base_url:
        dataset = load_dataset("parquet", data_files={'train': [
                            #base_url+'train-0.parquet', base_url+'train-1.parquet', base_url+'train-2.parquet', base_url+'train-3.parquet', base_url+'train-4.parquet', base_url+'train-5.parquet', base_url+'train-6.parquet', base_url+'train-7.parquet', base_url+'train-8.parquet'], 
                            base_url+'train-9.parquet', base_url+'train-10.parquet', base_url+'train-11.parquet', base_url+'train-12.parquet', base_url+'train-13.parquet', base_url+'train-14.parquet', base_url+'train-15.parquet', base_url+'train-16.parquet', base_url+'train-17.parquet'], 
                            'validation': [base_url+'validation-0.parquet', base_url+'validation-1.parquet', base_url+'validation-2.parquet'],
                            'test': [base_url+'test-0.parquet', base_url+'test-1.parquet', base_url+'test-2.parquet']})
    if 'metaqa' in base_url:
        dataset = load_dataset('json', data_files={'train':[base_url+'train.json'], 'validation':[base_url+'validation.json'], 'test':[base_url+'test.json']})
    
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    answer = dataset['test']['answer']
    return train_dataset, val_dataset, test_dataset

def postprocess(path):
    lst = str([tuple(inner_list) for inner_list in path])[1:-1]
    return lst

def triple2yaml(path):
    if isinstance(path[0], str):
        knowledge_graph_tuples = [eval(t) for t in path]
    else:
        knowledge_graph_tuples = path
    knowledge_graph = {}
    for subject, predicate, obj in knowledge_graph_tuples:
        keys = predicate.split('.')
        current_level = knowledge_graph.setdefault(subject, {})
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        if keys[-1] not in current_level:
            current_level[keys[-1]] = obj
        else:
            if isinstance(current_level[keys[-1]], list):
                current_level[keys[-1]].append(obj)
            else:
                current_level[keys[-1]] = [current_level[keys[-1]], obj]
    knowledge_yaml = yaml.dump(knowledge_graph, default_flow_Fstyle=False, sort_keys=False)
    return knowledge_yaml



def shortest_path(data_path):
    dict = {}
    with open(data_path, 'r') as json_file:
        for line in json_file:
            line = json.loads(line)
            question=line['question']
            answer=line['a_entity']
            graph = line['paths']
            if len(graph)==0:
                continue
            else:
                shortest_length = max(len(lst) for lst in graph)
                shortest_lists = [lst for lst in graph if len(lst) == shortest_length]
            dict[question] = shortest_lists
            
            
    return dict
                

def get_final_query(query, path, answer):
    task_description = "Here is a problem, along with clues from a knowledge graph and the answer. Please provide the corresponding reasoning process.\n\n"

    icl = """Here are some examples:
Input: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
Clues: ("George Washington Colonials men's basketball", 'sports.sports_team.arena_stadium', 'Charles E. Smith Center'), ('Charles E. Smith Center', 'location.location.containedby', 'Washington, D.C.')
Answer: ['Washington, D.C.']
Output: Step 1: Identify the Home Arena of the Team. The home arena for the George Washington Colonials men's basketball team is the Charles E. Smith Center.
Step 2: Locate the Arena. The Charles E. Smith Center is contained by, or located within, Washington, D.C.

Input: What year did the team with Baltimore Fight Song win the Superbowl?
Clues: ('Super Bowl', 'sports.sports_championship.events', 'Super Bowl XLVII')
Answer: ['Super Bowl XLVII']
Output: Step 1: Identify the Team Associated with the Baltimore Fight Song. The Baltimore Fight Song is associated with the Baltimore Ravens NFL team.
Step 2: Determine the Super Bowl Victory of the Team. The Baltimore Ravens won Super Bowl XLVII.

Input: What movie with film character named Mr. Woodson did Tupac star in?
Clues: ('Mr. Woodson', 'film.film_character.portrayed_in_films', 'm.03jps9y'), ('m.03jps9y', 'film.performance.film', "Gridlock'd")
Answer: ["Gridlock'd"]
Output: Step 1: Identify the Film Character. Mr. Woodson is a character that appears in a film.
Step 2: Find the Film. The film character Mr. Woodson is portrayed in the film film "Gridlock'd".

"""

    query = task_description + icl +'Input: ' + query +'\nClues: ' + path + '\nAnswer: ' + answer + '\nOutput: '

    return query


def inference(model, sample, path):
    query = sample["question"]
    q_entity = sample["q_entity"]
    a_entity = sample['a_entity']
    
    
    query = get_final_query(query,path,str(a_entity))
    
    with torch.no_grad():
        sampling_params = SamplingParams(max_tokens=512, use_beam_search=True, n=5, temperature=0)
        outputs = model.generate(query, sampling_params)
        response = outputs[0].outputs[0].text
    
    
    response = response.strip().strip('#').strip('\n').split('\nInput')[0].strip('\n')

    #print(query)      
    print(response)
    exit()
    return response


device = torch.device("cuda")
model_path = '/ossfs/workspace/common_base_model/llama2-70b-chat'
data_path = '/ossfs/workspace/data/webqsp/'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if '70b' in model_path:
    model = LLM(model=model_path,tensor_parallel_size=2, seed=0)
else:
    model = LLM(model=model_path, seed=0)
train_data, val_data, test_data = load_data(data_path)


path_dict = shortest_path('/ossfs/workspace/data/webqsp_path_validation.jsonl')
prediction = []
for idx in range(len(val_data)):
    ques = val_data[idx]['question']
    if ques in path_dict.keys():
        paths = path_dict[ques]
    else:
        continue
    for path in paths:
        pathstr = postprocess(path)
        #task = "Please think step by step and then answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list. If there are hints, please combine this information to answer.\n\n"
        pre = inference(model, train_data[idx], pathstr)
        dict = {'question': train_data[idx]['question'], 'cot': pre, 'path': path}

        #dict = {'input':task+ques, 'output':pre+'\n### Output: '+}
    
        prediction.append(dict)
    if idx % 200 == 0:
        print("inference {} of {} samples...".format(idx, len(train_data)))
    

with open('/ossfs/workspace/webqsp_validation_cotstep.jsonl','w') as json_write:
    json.dump(prediction, json_write, indent=4)
    
