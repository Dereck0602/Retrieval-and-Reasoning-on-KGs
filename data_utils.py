from datasets import load_dataset
import json
from difflib import SequenceMatcher

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
        dataset = load_dataset('json', data_files={'train':[base_url+'train.json'],'test':[base_url+'test.json']})
    
    train_dataset = dataset['train']
    
    test_dataset = dataset['test']
    answer = dataset['test']['answer']
    '''
    with open('webqsp_answer.json', 'w') as json_file:
        for i in answer:
            #data = json.loads(i)
            #data = literal_eval(i)
            json.dump(i, json_file)
            json_file.write('\n')
    '''
    return train_dataset, test_dataset

def create_graph():
    base_url1='/ossfs/workspace/data/webqsp/'
    base_url2='/ossfs/workspace/data/webcwq/'
    dataset = load_dataset("parquet", data_files={'train': [base_url1+'train-1.parquet', base_url1+'train-2.parquet', base_url2+'train-0.parquet', base_url2+'train-1.parquet', base_url2+'train-2.parquet', base_url2+'train-3.parquet', base_url2+'train-4.parquet', base_url2+'train-5.parquet', base_url2+'train-6.parquet',
                            base_url2+'train-7.parquet', base_url2+'train-8.parquet', base_url2+'train-9.parquet', base_url2+'train-10.parquet', base_url2+'train-11.parquet', base_url2+'train-12.parquet', base_url2+'train-13.parquet', base_url2+'train-14.parquet', base_url2+'train-15.parquet', base_url2+'train-16.parquet', 
                            base_url2+'train-17.parquet', base_url1+'validation.parquet', base_url2+'validation-0.parquet', base_url2+'validation-1.parquet', base_url2+'validation-2.parquet', base_url1+'test-1.parquet', base_url1+'test-2.parquet', base_url2+'test-0.parquet', base_url2+'test-1.parquet', base_url2+'test-2.parquet']})
    graph_dict = {}
    for sample in dataset['train']:
        graph = sample['graph']
        for triple in graph:
            head, relation, tail = triple
            if head not in graph_dict:
                graph_dict[head] = []
            pair = (head, relation, tail)
            if pair not in graph_dict[head]:
                graph_dict[head].append(pair)
                
    with open('graph.json', 'w') as json_file:
        json.dump(graph_dict, json_file)
        #json_file.write('\n')
    #print(len(graph_dict))
    #print(len(graph_dict['P!nk']))
    #exit()


def create_metaqa_graph():
    dataset = load_dataset('text', data_files={'train':'/ossfs/workspace/data/metaqa/kb.txt'})['train']
    
    graph_dict = {}
    for sample in dataset:
        
        head, relation, tail = sample['text'].split('|')
        if head not in graph_dict:
            graph_dict[head] = []
        pair = (head, relation, tail)
        if pair not in graph_dict[head]:
            graph_dict[head].append(pair)
    
    with open('metaqa_graph.json', 'w') as json_file:
        json.dump(graph_dict, json_file)


def load_graph(graph):
    with open(graph, 'r') as json_file:
        fcc_data = json.load(json_file)
        #keys = list(fcc_data.keys())
        print(fcc_data['Yankel Murciano'])
        exit()
    return fcc_data

def get_keys(graph):
    return list(graph.keys())

def query_entity(query, keys, graph):
    #triple = []
    if query in keys:
        return graph[query]

     

    
