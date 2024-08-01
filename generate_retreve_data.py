import json
import yaml
import random

def load_graph(graph):
    with open(graph, 'r') as json_file:
        fcc_data = json.load(json_file)
        
    return fcc_data


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



def positive_sample(data_path, aux_triple=False, aux_data=None, use_yaml=False, topk=3):
    dict = {}
    with open(data_path, 'r') as json_file:
        lines = json.load(json_file)
        for line in lines:
            #line = json.loads(line)
            question=line['question']
            cot=line['cot'].split('\n')
            path = line['path']

            if aux_triple:
                #lst = [item for sublist in sampled_lists for item in sublist]
                sampled_lists = []
                for item in path:
                    h, r, t = item
                    if h in aux_data.keys() and t in aux_data.keys():
                        aux_h = aux_data[h]
                        aux_sample = min(3, len(aux_h))
                        aux_h = random.sample(aux_h, aux_sample)
                        aux_t = aux_data[t]
                        aux_sample = min(3, len(aux_t))
                        aux_t = random.sample(aux_t, aux_sample)
                        aux = aux_h+aux_t
                    elif h in aux_data.keys() and t not in aux_data.keys():
                        aux = aux_data[h]
                        aux_sample = min(6, len(aux))
                        aux = random.sample(aux, aux_sample)
                    elif t in aux_data.keys() and h not in aux_data.keys():
                        aux = aux_data[t]
                        aux_sample = min(6, len(aux))
                        aux = random.sample(aux, aux_sample)
                    
                    if use_yaml:
                        aux=[tuple(i) for i in aux]
                        aux.append(tuple(item))
                        aux = triple2yaml(aux)
                        item = str(tuple(item))+'\nSubgraph:\n'+aux
                        sampled_lists.append(item)
                    else:
                        aux=str([tuple(i) for i in aux])[1:-1]
                        sampled_lists.append(str(tuple(item))+'\nNeighbor triples: '+aux)
                    #print(sampled_lists)
                    #exit()
                        
            else:
                sampled_lists = [str(tuple(item)) for item in path]
            
            if len(sampled_lists)==len(cot):
                for i in range(len(cot)):
                    question = question + '\n'+cot[i]
                    dict[question]=sampled_lists[:i+1]
                    #print(dict)
                    #exit()
            elif len(sampled_lists)<len(cot):
                for i in range(len(sampled_lists)):
                    question = question + '\n'+cot[i]
                    dict[question]=sampled_lists[:i+1]
                for i in range(len(sampled_lists), len(cot)):
                    question = question + '\n'+cot[i]
                    dict[question]=sampled_lists
            else:
                for i in range(len(cot)):
                    question = question + '\n'+cot[i]
                    dict[question]=sampled_lists[:i+1]
                dict[question]=sampled_lists
                
    return dict
            

def sample(data_path, positive_sample, aux_triple=False, aux_data=None, use_yaml=False, negative_num=5):
    num=0
    lst = []
    random_negative = []
    all_graph = {}
    with open(data_path, 'r') as json_file:
        fcc_data = json.load(json_file)
        for line in fcc_data:
            qes = line['question']
            all = line['graph']
            all_graph[qes]=all
            

    for sample, positive in positive_sample.items():
        positive_item = [i.split('\nNeighbor')[0] for i in positive]
        query = sample.split('\n')[0]
        all_triple = all_graph[query]
        negative = list(set(all)-set(positive_item))
        num+=1

        for i in positive:
            if len(negative)>negative_num:
                neg = random.sample(negative, negative_num)
            else:
                neg = negative+random.sample(random_negative, negative_num-len(negative))

            if aux_triple:
                neg_lst = []
                for neg_item in neg:
                    h, r, t = eval(neg_item)
                    #h=h.strip()[1:-1]
                    #t=t.strip()[1:-1]
                    if h in aux_data.keys() and t in aux_data.keys():
                        aux_h = aux_data[h]
                        aux_sample = min(3, len(aux_h))
                        aux_h = random.sample(aux_h, aux_sample)
                        aux_t = aux_data[t]
                        aux_sample = min(3, len(aux_t))
                        aux_t = random.sample(aux_t, aux_sample)
                        aux = aux_h+aux_t
                    elif h in aux_data.keys() and t not in aux_data.keys():
                        aux = aux_data[h]
                        aux_sample = min(6, len(aux))
                        aux = random.sample(aux, aux_sample)
                    elif t in aux_data.keys() and h not in aux_data.keys():
                        aux = aux_data[t]
                        aux_sample = min(6, len(aux))
                        aux = random.sample(aux, aux_sample)

                    if use_yaml:
                        aux=[tuple(i) for i in aux]
                        aux.append(eval(neg_item))
                        
                        aux = triple2yaml(aux)
                        item = str(tuple(neg_item))+'\nSubgraph:\n'+aux
                        neg_lst.append(item)
                    else:
                        aux=str([tuple(i) for i in aux])[1:-1]
                        neg_lst.append(neg_item+'\nNeighbor triples: '+aux)
                dict={"query":sample, "pos": [i], "neg": neg_lst}
                #print(dict)
            else:        
                dict={"query":sample, "pos": [i], "neg": neg}
            lst.append(dict)
        random_negative+=all_triple
        
                
        if num%500==0:
            print('Complete ', num)
    with open('/ossfs/workspace/data/kg_data/webcwq_cot_retrieve_kg_neighbor_2.jsonl', 'w') as f:
        for entry in lst:
            f.write(json.dumps(entry) + '\n')



graph = load_graph('graph.json')
positive = positive_sample('/ossfs/workspace/webcwq_train_cotstep_2.jsonl', aux_triple=True, aux_data=graph, use_yaml=False, topk=1)
sample('/ossfs/workspace/webcwq_retrievekg_embedder_top50_train_2.jsonl', positive, aux_triple=True, aux_data=graph, use_yaml=False)
