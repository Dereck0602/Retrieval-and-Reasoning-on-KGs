import json
import yaml
import random
import os
import copy
from datasets import load_dataset

def triple2json(path):
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
    knowledge_json = json.dumps(knowledge_graph, indent=2)
    return knowledge_json


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

def preprocess_relation(tup):
    head, rela, tail = tup
    rela = rela.split('.')[-1]
    return (head,rela,tail)

def postprocess(path):
    lst = str([tuple(inner_list) for inner_list in path])[1:-1]
    
    return lst

def entity_instruction(instruction, entity, neighbor, k, use_json=False, use_yaml=False):
    ## predict head entity ##
    """
    input: entity and its 1-hop neighbor
    return k instruction
    """
    
    k=min(k, len(neighbor))
    rela = random.sample(neighbor, k)
    if use_json:
        rela=triple2json(rela)
    elif use_yaml:
        rela=triple2yaml(rela)
    else:
        rela = postprocess(rela)+'\n'
    data = instruction+'Input: '+rela
    
    dict = {"input": data, 'output': entity}
    return dict


def sample(lst, k):
    if k>len(lst):
        k=len(lst)
    return random.sample(lst, k)

def recog_entity(path):
    head = path[0][0]
    tail = path[-1][-1]
    return head, tail

def shuffle_path(path):
    shuffle_path = copy.deepcopy(path)
    random.shuffle(shuffle_path)
    return shuffle_path

def negative_triple(graph,path, ratio):
    nagetive_path = copy.deepcopy(path)
    mention_entity = [s[0] for s in path]
    
    for entity in mention_entity:
        triple = random.choices(graph[entity],k=1)
        if triple not in path:
            nagetive_path.append(triple)
    return nagetive_path

def delete_triple(path):
    path = shuffle_path(path)
    delete_path = path[1:]
    return delete_path



def create_entity_data(use_json=False, use_yaml=False):
    instruction1 = 'Please predict the entity represented by <mask> based on the one-hop relationships in the knowledge graph.\n'
    instruction2 = 'Based on the one-hop relationships in the knowledge graph, infer the entity represented by <mask>.\n'
    instruction3 = 'Make a prediction about the masked entity, using the one-hop relationships in the knowledge graph as a reference.\n'
    instruction = [instruction1, instruction2, instruction3]

    with open('/ossfs/workspace/graph.json', 'r') as json_file:      
        lst = []
        graph = json.load(json_file)
        entity = graph.keys()
        entity_lst = [s for s in entity if not s.startswith('m.') and len(graph[s])>5]
        for i in range(len(entity_lst)):
            neighbor = graph[entity_lst[i]]
            neighbor = [sub_lst for sub_lst in neighbor if not any(item.startswith('m.') for item in sub_lst)]
            neighbor = [['<mask>'] + sub_lst[1:] for sub_lst in neighbor]
            if len(neighbor)==0:
                continue
            else:
                instr = instruction[i%3]
                lst.append(entity_instruction(instr, entity_lst[i], neighbor, 5, use_json=use_json, use_yaml=use_yaml))

    if use_yaml:
        with open('/ossfs/workspace/data/entity_instruction_yaml.json','w') as json_write:
            json.dump(lst, json_write, indent=4)
    elif use_json:
        with open('/ossfs/workspace/data/instruction_json.json','w') as json_write:
            json.dump(lst, json_write, indent=4)
    else:
        with open('/ossfs/workspace/data/instruction_triple.json','w') as json_write:
            json.dump(lst, json_write, indent=4)


def create_relationship_data(maxk=2, use_yaml=False):
    instruction1 = 'Please recognize the relationship between the two entities.\nKnowledge Graph: '
    instruction2 = 'Please predict the relationship between the two entities.\nThere are some one-hop information of these entities: '
    instruction3 = 'Make a prediction about the relationship, using the one-hop relationships in the knowledge graph as a reference.\n'
    instruction = [instruction1, instruction2, instruction3]
    with open('/ossfs/workspace/graph.json', 'r') as json_file:      
        lst = []
        graph = json.load(json_file)
        entity = graph.keys()
        entity = [s for s in entity if not s.startswith('m.') and len(graph[s])>5]
        triple_lst = []
        for i in entity:
            triple = [s for s in graph[i] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
            triple = sample(triple, maxk)
            triple_lst+=triple
        
        for i in range(len(triple_lst)):
            triple = triple_lst[i]
            head, relation, tail = triple
            if i%4==0 or i%4==1:
                instr = instruction[0]
                if head not in graph.keys():
                    head_neighbor = []
                else:
                    head_neighbor = [s for s in graph[head] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
                    head_neighbor = sample(head_neighbor, 5)
                if tail not in graph.keys():
                    tail_neighbor = []
                else:
                    tail_neighbor = [s for s in graph[tail] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
                    tail_neighbor = sample(tail_neighbor, 5)
                neighbor = head_neighbor + tail_neighbor
                neighbor.append(triple)
                if use_yaml:
                    neighbor = '\n'+triple2yaml(neighbor)
                else:
                    neighbor = postprocess(neighbor)+'\n'
                data = instr+str(neighbor)+'Input: '+'\''+head+'\''+' and '+'\''+tail+'\''+'\n'
                
                dict = {"input": data, 'output': relation}
                lst.append(dict)
            elif i%4==2:
                instr = instruction[1]
                if head not in graph.keys():
                    head_neighbor = []
                else:
                    head_neighbor = [s for s in graph[head] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
                    head_neighbor = sample(head_neighbor, 5)
                if tail not in graph.keys():
                    tail_neighbor = []
                else:
                    tail_neighbor = [s for s in graph[tail] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
                    tail_neighbor = sample(tail_neighbor, 5)
                neighbor = head_neighbor + tail_neighbor
                if use_yaml:
                    neighbor = '\n'+triple2yaml(neighbor)
                else:
                    neighbor = postprocess(neighbor)+'\n'
                data = instr+str(neighbor)+'Input: '+'\''+head+'\''+' and '+'\''+tail+'\''+'\n'
                
                dict = {"input": data, 'output': relation}
                lst.append(dict)
            else:
                instr = instruction[2]
                if head not in graph.keys():
                    head_neighbor = []
                else:
                    head_neighbor = [s for s in graph[head] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
                    head_neighbor = sample(head_neighbor, 5)
                if tail not in graph.keys():
                    tail_neighbor = []
                else:
                    tail_neighbor = [s for s in graph[tail] if not (s[0].startswith('m.') or s[-1].startswith('m.'))]
                    tail_neighbor = sample(tail_neighbor, 5)
                neighbor = head_neighbor + tail_neighbor
                if use_yaml:
                    neighbor = '\n'+triple2yaml(neighbor)
                else:
                    neighbor = postprocess(neighbor)+'\n'
                data = instr+'One-hot Information: '+str(neighbor)+'Input: '+'\''+head+'\''+' and '+'\''+tail+'\''+'\n'
                dict = {"input": data, 'output': relation}
                lst.append(dict)
        print(len(lst))
    with open('/ossfs/workspace/data/relation_instruction_yaml.json','w') as json_write:
        json.dump(lst, json_write, indent=4)



def create_graph2text(data_dir, use_json=False, use_yaml=False):
    instruction1 = 'Please deeply understand the following knowledge graph, and then convert them into a coherent sentence\nInput: '
    instruction2 = 'Given these knowledge graph, please deeply write a paragraph that integrates the information contained in them\nInput: '
    instruction3 = 'Please deeply understand the following knowledge graph, and then generate an explanatory text that connects these knowledge graph triples\nInput: '
    instruction4 = 'Compose an informative report using the information from these knowledge graph\nInput: '
    instruction = [instruction1, instruction2, instruction3, instruction4]
    data = load_dataset('parquet', data_files={'train':data_dir+'train.parquet', 'validation':data_dir+'dev.parquet'})
    lst = []
    num = 0
    for example in data['train']:
        text = example['text']
        triple = example['triplets']
        triple_lst = [(i.split('|')[0], i.split('|')[1], i.split('|')[2]) for i in triple]
        if use_json:
            graph = triple2json(triple_lst)
        elif use_yaml:
            graph = triple2yaml(triple_lst)
        else:
            graph = str(triple_lst)[1:-1]+'\n'
        
        instr = instruction[num%4]
        num += 1
        dict = {'input': instr+'\n'+graph, 'output':text}
        
        lst.append(dict)

    for example in data['validation']:
        text = example['text']
        triple = example['triplets']
        triple_lst = [(i.split('|')[0], i.split('|')[1], i.split('|')[2]) for i in triple]
        if use_json:
            graph = triple2json(triple_lst)
        elif use_yaml:
            graph = triple2yaml(triple_lst)
        else:
            graph = str(triple_lst)[1:-1]+'\n'
        instr = instruction[num%4]
        num += 1
        dict = {'input': instr+'\n'+graph, 'output':text}
        lst.append(dict)
    #print(data['train'][0])

    with open('/ossfs/workspace/data/graph2text_instruction_yaml.json','w') as json_write:
        json.dump(lst, json_write, indent=4)


def create_text2graph(data_dir, use_json=False, use_yaml=False):
    instruction1 = 'Please extract all entities and relationships in the sentence.\nInput: '
    instruction2 = 'Given the sentence, please extract a knowledge graph that integrates the information contained in them\nInput: '
    instruction3 = 'Please deeply understand the following sentence, and then generate a knowledge graph\nInput: '
    instruction = [instruction1, instruction2, instruction3]
    data = load_dataset('parquet', data_files={'train':data_dir+'train.parquet', 'validation':data_dir+'dev.parquet'})
    print(data)
    exit()
    lst = []
    num = 0
    for example in data['train']:
        text = example['text']
        triple = example['triplets']
        triple_lst = [(i.split('|')[0], i.split('|')[1], i.split('|')[2]) for i in triple]
        if use_json:
            graph = triple2json(triple_lst)
        elif use_yaml:
            graph = triple2yaml(triple_lst)
        else:
            graph = str(triple_lst)[1:-1]+'\n'
        
        instr = instruction[num%3]
        num += 1
        dict = {'input': instr+text, 'output':graph}
        
        
        lst.append(dict)

    for example in data['validation']:
        text = example['text']
        triple = example['triplets']
        triple_lst = [(i.split('|')[0], i.split('|')[1], i.split('|')[2]) for i in triple]
        if use_json:
            graph = triple2json(triple_lst)
        elif use_yaml:
            graph = triple2yaml(triple_lst)
        else:
            graph = str(triple_lst)[1:-1]+'\n'
        instr = instruction[num%3]
        num += 1
        dict = {'input': instr+text, 'output':graph}
        lst.append(dict)
    #print(data['train'][0])

    with open('/ossfs/workspace/data/text2graph_instruction_yaml.json','w') as json_write:
        json.dump(lst, json_write, indent=4)

create_entity_data(use_yaml=True)
create_relationship_data(use_yaml=True)
create_graph2text('/ossfs/workspace/data/webnlg/',use_yaml=True)
create_text2graph('/ossfs/workspace/data/webnlg/',use_yaml=True)
