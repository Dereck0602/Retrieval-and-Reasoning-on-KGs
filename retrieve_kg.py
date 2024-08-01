import os
import random
import torch
from datasets import load_dataset
import json
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
        dataset = load_dataset('json', data_files={'train':[base_url+'train.json'],'validation':[base_url+'validation.json'],'test':[base_url+'test.json']})
    
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    answer = dataset['test']['answer']
    return train_dataset, val_dataset, test_dataset


def load_graph(graph):
    with open(graph, 'r') as json_file:
        fcc_data = json.load(json_file)     
    return fcc_data


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
    knowledge_yaml = yaml.dump(knowledge_graph, default_flow_style=False, sort_keys=False)
    return knowledge_yaml


def retrieve(dataset, model, save_path, graph=None, use_yaml=False, topk=10):
    lst = []
    num=0
    #with open('/ossfs/workspace/webcwq_llama2_7b_chat_cot_output.json', 'r') as file:
    with open('/ossfs/workspace/webqsp_woentity_cot_output.json', 'r') as file:
        cot = json.load(file)
        
    instruction={"query": "Represent this query for retrieving relevant knowledge graph triples: ",
        "key": "Represent this knowledge graph triple for retrieval: "}
    for sample in dataset:
        #question = [instruction['query']+sample['question']+' CoT: '+cot[num]['cot']]
        #question = [instruction['query']+sample['question']]
        CoT = cot[num]['cot'].strip('\n').split('\n')
        
        question = [sample['question']]
        #candidate = [instruction['key']+str(tuple(i)) for i in sample['graph'] if 'sentence' not in i[1]]
        candidate = [str(tuple(i)) for i in sample['graph'] if 'sentence' not in i[1]]
        
        if graph:
            aux_lst = []
            for item in candidate:
                h, r, t = eval(item)
                if h in graph.keys() and t in graph.keys():
                    aux_h = graph[h]
                    aux_sample = min(3, len(aux_h))
                    aux_h = random.sample(aux_h, aux_sample)
                    aux_t = graph[t]
                    aux_sample = min(3, len(aux_t))
                    aux_t = random.sample(aux_t, aux_sample)
                    aux = aux_h+aux_t
                elif h in graph.keys() and t not in graph.keys():
                    aux = graph[h]
                    aux_sample = min(6, len(aux))
                    aux = random.sample(aux, aux_sample)
                elif t in graph.keys() and h not in graph.keys():
                    aux = graph[t]
                    aux_sample = min(6, len(aux))
                    aux = random.sample(aux, aux_sample)
                

                if use_yaml:
                    aux=[tuple(i) for i in aux]
                    aux.append(eval(item))
                    aux = triple2yaml(aux)
                    item = str(eval(item))+'\nSubgraph:\n'+aux
                    aux_lst.append(item)
                else:
                    aux=str([tuple(i) for i in aux])[1:-1]
                    aux_lst.append(item+'\nNeighbor triples: '+aux)
                
            candidate = aux_lst
        
        if len(candidate)!=0:
            embeddings_2 = model.encode(candidate, normalize_embeddings=True)

            all_similarity = []
            embeddings_1 = model.encode(question, normalize_embeddings=True)
            similarity = embeddings_1 @ embeddings_2.T
            similarity_scores = similarity.flatten()

            
            all_similarity.append(similarity_scores)
            
            for i in CoT:
                question = [question[0] + '\n'+i]
                #print(question)
                embeddings_1 = model.encode(question, normalize_embeddings=True)
                similarity = embeddings_1 @ embeddings_2.T
                similarity_scores = similarity.flatten()
                all_similarity.append(similarity_scores)
            arr = np.arange(1, len(all_similarity)+1, 1)
            
            similarity_scores = np.average(np.stack(all_similarity, axis=0), weights=arr, axis=0)
            

            sorted_indices = np.argsort(similarity_scores)[::-1]
            topk_indices = sorted_indices[:topk]  ## top 10
            #topk_candidates = [candidate[i].replace(instruction['key'],'',1) for i in topk_indices]
            if graph:
                if use_yaml:
                    topk_candidates = [candidate[i].split('\nSubgraph:\n')[0] for i in topk_indices]
                    aux_graph = [candidate[i].split('\nSubgraph:\n')[1] for i in topk_indices]
                else:
                    topk_candidates = [candidate[i].split('\nNeighbor triples: ')[0] for i in topk_indices]
                    aux_graph = [candidate[i].split('\nNeighbor triples: ')[1] for i in topk_indices]
            else:
                topk_candidates = [candidate[i] for i in topk_indices]
                
            
            score = [similarity_scores[i] for i in topk_indices]
            if graph:
                dict = {'question':sample['question'], 'graph': topk_candidates, 'answer': sample['answer'], 'score':str(score), 'aux_graph': aux_graph}
                
            else:
                dict = {'question':sample['question'], 'graph': topk_candidates, 'answer': sample['answer'], 'score':str(score)}
            
            lst.append(dict)
            
            num+=1
        if num%500==0:
            print("Complete :", num)
    with open(save_path,'w') as json_write:
        json.dump(lst, json_write, indent=4)


def meta_retrieve(dataset, model, save_path, graph=None, use_yaml=False, topk=10):
    lst = []
    num=0
    with open('/ossfs/workspace/metaqa3hop_continue_instruct_qacot_cot_output.json', 'r') as file:
        cot = json.load(file)
        
    instruction={"query": "Represent this query for retrieving relevant knowledge graph triples: ",
        "key": "Represent this knowledge graph triple for retrieval: "}

    graph_dataset = load_dataset('text', data_files={'train':'/ossfs/workspace/data/metaqa/kb.txt'})['train']
    candidate = []
    for i in graph_dataset:
        head, relation, tail = i['text'].split('|')
        candidate.append(str((head, relation, tail)))
    
    if graph:
        aux_lst = []
        for item in candidate:
            h, r, t = eval(item)
            if h in graph.keys() and t in graph.keys():
                aux_h = graph[h]
                aux_sample = min(3, len(aux_h))
                aux_h = random.sample(aux_h, aux_sample)
                aux_t = graph[t]
                aux_sample = min(3, len(aux_t))
                aux_t = random.sample(aux_t, aux_sample)
                aux = aux_h+aux_t
            elif h in graph.keys() and t not in graph.keys():
                aux = graph[h]
                aux_sample = min(6, len(aux))
                aux = random.sample(aux, aux_sample)
            elif t in graph.keys() and h not in graph.keys():
                aux = graph[t]
                aux_sample = min(6, len(aux))
                aux = random.sample(aux, aux_sample)
            
            if use_yaml:
                aux=[tuple(i) for i in aux]
                aux.append(eval(item))
                aux = triple2yaml(aux)
                item = str(eval(item))+'\nSubgraph:\n'+aux
                aux_lst.append(item)
            else:
                aux=str([tuple(i) for i in aux])[1:-1]
                aux_lst.append(item+'\nNeighbor triples: '+aux)   
        candidate = aux_lst

    embeddings_2 = model.encode(candidate, normalize_embeddings=True)

    for sample in dataset:
        #question = [instruction['query']+sample['question']+' CoT: '+cot[num]['cot']]
        #question = [instruction['query']+sample['question']]
        CoT = cot[num]['cot'].strip('\n').split('\n')
        question = [sample['question']]
        
        all_similarity = []
        embeddings_1 = model.encode(question, normalize_embeddings=True)
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores = similarity.flatten()

        
        all_similarity.append(similarity_scores)
        for i in CoT:
            question = [question[0] + '\n'+i]
            #print(question)
            embeddings_1 = model.encode(question, normalize_embeddings=True)
            similarity = embeddings_1 @ embeddings_2.T
            similarity_scores = similarity.flatten()
            all_similarity.append(similarity_scores)
        arr = np.arange(1, len(all_similarity)+1, 1)
        similarity_scores = np.average(np.stack(all_similarity, axis=0), weights=arr, axis=0)
        

        sorted_indices = np.argsort(similarity_scores)[::-1]
        topk_indices = sorted_indices[:topk]  ## top 10
        #topk_candidates = [candidate[i].replace(instruction['key'],'',1) for i in topk_indices]
        if graph:
            if use_yaml:
                topk_candidates = [candidate[i].split('\nSubgraph:\n')[0] for i in topk_indices]
                aux_graph = [candidate[i].split('\nSubgraph:\n')[1] for i in topk_indices]
            else:
                topk_candidates = [candidate[i].split('\nNeighbor triples: ')[0] for i in topk_indices]
                aux_graph = [candidate[i].split('\nNeighbor triples: ')[1] for i in topk_indices]
        else:
            topk_candidates = [candidate[i] for i in topk_indices]
            
        
        score = [similarity_scores[i] for i in topk_indices]
        if graph:
            dict = {'question':sample['question'], 'graph': topk_candidates, 'answer': sample['answer'], 'score':str(score), 'aux_graph': aux_graph}
            
        else:
            dict = {'question':sample['question'], 'graph': topk_candidates, 'answer': sample['answer'], 'score':str(score)}
        
        lst.append(dict)
        
        num+=1
        if num%500==0:
            print("Complete :", num)
    with open(save_path,'w') as json_write:
        json.dump(lst, json_write, indent=4)


graph = load_graph('graph.json')
train,val,test=load_data('/ossfs/workspace/data/webqsp/')
model = SentenceTransformer('/ossfs/workspace/model/retrieve_kg_cot_neighbor_model', device='cuda')
retrieve(test, model, '/ossfs/workspace/webqsp_retrievekg_top50.jsonl', graph=graph, use_yaml=False, topk=50)
#sentences_1 = ["what is the name of justin bieber brother?"]
#sentences_2 = ["(Justin Bieber, has a half-brother, Jaxon Bieber)", "(Jaxon Bieber, is the son of, Jeremy Bieber)", "(Jeremy Bieber, has a second wife, Erin Wagner)", "(Jaxon Bieber, was born on, November 20, 2009)"]
