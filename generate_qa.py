import torch
import random
import re
import json
import yaml
from ast import literal_eval
from vllm import LLM, SamplingParams

def postprocess(path):
    lst = str([tuple(inner_list) for inner_list in path])[1:-1]
    return lst

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


class GenQA:
    """
    Inference code for DAIL. You can inference your data with two steps:
    1). Init:             inferencer = DAIL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, tokenizer, model, train_dataset, test_dataset, device, case_study=False, sentence_model=None, embedding_bank=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.sentence_model = sentence_model
        self.embedding_bank = embedding_bank
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.case_study = case_study

    def get_embedding(self, sentence):
        embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
        return embedding
        
    def get_final_query(self, query, link=None, cot=None):
        if self.args.code_prompt:
            task_description = "Please think step by step and then complete the yaml file.\n\n" #
        else:
            task_description = "Please think step by step and then answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list. If there are hints, please combine this information to answer.\n\n" # 
        if self.args.prompt == 'cot':
            if link:
                graph1 = [('m.0gxnnwp', 'people.sibling_relationship.sibling', 'Justin Bieber'), ('Justin Bieber', 'people.person.sibling_s', 'm.0gxnnwc'), ('Justin Bieber', 'people.person.sibling_s', 'm.0gxnnwp'), ('m.0gxnnwc', 'people.sibling_relationship.sibling', 'Justin Bieber'), ('Jaxon Bieber', 'people.person.sibling_s', 'm.0gxnnwp')]
                graph2 = [('Grand Bahama', 'location.statistical_region.population', 'm.0hvqzry'), ('West Grand Bahama', 'location.administrative_division.first_level_division_of', 'Bahamas'), ('Grand Bahama', 'location.location.containedby', 'Bahamas'), ('Grand Bahama', 'location.location.people_born_here', 'Juan Lewis'), ('East Grand Bahama', 'location.administrative_division.first_level_division_of', 'Bahamas')]
                graph3 = [('Washington Redskins', 'sports.sports_team.league', 'm.0crt504'), ('Washington Redskins', 'sports.sports_team.location', 'Washington, D.C.'), ('m.0793qh6', 'american_football.game_receiving_statistics.team', 'Washington Redskins'), ('Washington Redskins', 'sports.sports_team.league', 'm.0crt9bx'), ('Boston Redskins', 'sports.sports_team.league', 'm.0crt572')]
                
                #graph1 = [('m.0wz2x7d', 'sports.team_venue_relationship.team', "George Washington Colonials men's basketball"), ("George Washington Colonials men's basketball", 'sports.sports_team.venue', 'm.0wz2x7d'), ("George Washington Colonials men's basketball", 'sports.sports_team.arena_stadium', 'Charles E. Smith Center'), ('m.05g30bf', 'base.marchmadness.ncaa_tournament_seed.team', "George Washington Colonials men's basketball"), ("George Washington Colonials men's basketball", 'sports.school_sports_team.school', 'George Washington University')]
                #graph2 = [('Ravenation', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.fight_song', 'For Baltimore'), ('Baltimore Ravens', 'sports.sports_team.fight_song', 'Ravenation'), ('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.fight_song', 'The Baltimore Fight Song')]
                #graph3 = [('Georgia', 'government.governmental_jurisdiction.governing_officials', 'm.0jsjnfj'), ('m.0jsjnfj', 'government.government_position_held.office_holder', 'Nathan Deal'), ('Georgia Republican primary, 2012', 'freebase.valuenotation.is_reviewed', 'End date'), ('Georgia Republican primary, 2012', 'freebase.valuenotation.is_reviewed', 'Start date'), ('Georgia gubernatorial election, 2010', 'common.topic.notable_for', 'g.12556r3lf')]
                if self.args.graph_with_yaml:
                    graph1 = triple2yaml(graph1)
                    graph2 = triple2yaml(graph2)
                    graph3 = triple2yaml(graph3)
                elif self.args.graph_with_json:
                    graph1 = triple2json(graph1)
                    graph2 = triple2json(graph2)
                    graph3 = triple2json(graph3)
                else:
                    graph1 = str(graph1)[1:-1]
                    graph2 = str(graph2)[1:-1]
                    graph3 = str(graph3)[1:-1]
                #link = postprocess(link)
                
                if not self.args.rethink:
                    
                    icl = """Here are some examples:
Input: what is the name of justin bieber brother?
Hints:\n"""+graph1+"""
CoT: Let's think step by step. First 'm.0gxnnwc' and 'm.0gxnnwp' are identifiers for Justin Bieber's siblings. 'Jaxon Bieber' is associated with 'm.0gxnnwp'. Thus, Jaxon Bieber is Justin Bieber's brother.
### Output: ['Jaxon Bieber']

Input: what country is the grand bahama island in?
Hints:\n"""+graph2+"""
CoT: Let's think step by step. Look for hints that explicitly mention a country in relation to Grand Bahama. The hint 'Grand Bahama' mentions it is 'containedby' the Bahamas. Other hints about population and geolocation do not directly answer the country. Therefore, the answer is "Bahamas".
### Output: ['Bahamas']

Input: where are the nfl redskins from?
Hints:\n"""+graph3+"""
CoT: Let's think step by step. The hints provided reference the "Washington Redskins" as the team in question. The hints also mention that the "Washington Redskins" are associated with the location "Washington, D.C." and with certain league identifiers. So, the answer is Washington, D.C.
### Output: ['Washington, D.C.']\n
"""
                    
                    '''
                    icl = """Here are some examples:
Input: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
Hints:\n"""+graph1+"""
CoT: Let's think step by step. Step 1: Identify the university associated with "George Washington Colonials men's basketball". From the hints provided, we have the following relevant information: ("George Washington Colonials men's basketball", 'sports.school_sports_team.school', 'George Washington University'). This indicates that the "George Washington Colonials men's basketball" team is associated with "George Washington University." Step 2: Determine the state where George Washington University is located. Although not explicitly provided in the hints, general knowledge tells us that George Washington University is located in Washington, D.C.
### Output: ['Washington, D.C.']

Input: What year did the team with Baltimore Fight Song win the Superbowl?
Hints:\n"""+graph2+"""
CoT: Let's think step by step. 1. Identify the team associated with the Baltimore Fight Song: The Baltimore Ravens.\n2. Determine the years the Baltimore Ravens won the Super Bowl: The Ravens won the Super Bowl in the 2000 and 2012 seasons.\n3. List the years corresponding to those seasons: The 2000 season's Super Bowl was in 2001 (Super Bowl XXXV), and the 2012 season's Super Bowl was in 2013 (Super Bowl XLVII).
### Output: ['Super Bowl XLVII', 'Super Bowl XXXV']

Input: Which governor of Georgia in 2011 is the politician whose tenure started last?
Hints:\n"""+graph3+"""
CoT: Let's think step by step.\nStep 1: Identify the governor of Georgia in 2011. The first hint gives us a connection between Georgia and a governing official with identifier 'm.0jsjnfj'.\nStep 2: Connect the identifier to a person. The second hint connects 'm.0jsjnfj' to an office holder named Nathan Deal.\nStep 3: Consider the election information. The third and fourth hints mention the Georgia Republican primary in 2012, indicating that there was a political event related to the governor's position after 2011.
### Output: ['Nathan Deal']\n
"""
                    '''

                    if 'webqsp' in self.args.dataset or 'metaqa' in self.args.dataset:
                        query = task_description + icl +'Input: ' + query + '?\nHints:\n'+link+'\nCoT: Let\'s think step by step.'
                    else:
                        query = task_description + icl +'Input: ' + query + '\nHints:\n'+link+'\nCoT: Let\'s think step by step.'
                    
                else:
                    
                    icl = """Here are some examples:
Input: what is the name of justin bieber brother?
CoT: Let's think step by step. Justin Bieber has a half-brother from his father Jeremy Bieber's side. His name is Jaxon Bieber. Jaxon is significantly younger than Justin and has occasionally appeared with his brother at events and on Justin's social media channels. Thus, Jaxon Bieber is Justin Bieber's brother.
Hints:\n"""+graph1+"""
Rethink: Rethink it in the context of hints and previous response. First 'm.0gxnnwc' and 'm.0gxnnwp' are identifiers for Justin Bieber's siblings. 'Jaxon Bieber' is associated with 'm.0gxnnwp'. Thus, Jaxon Bieber is Justin Bieber's brother.
### Output: ['Jaxon Bieber']

Input: what country is the grand bahama island in?
CoT: Let's think step by step. Step 1: Identify the region referred to by "Grand Bahama Island." Grand Bahama Island is one of the islands in the archipelago known as the Bahamas. Step 2: Conclude which country Grand Bahama Island is a part of. Since Grand Bahama Island is one of the islands in the Bahamas archipelago and the Bahamas is an independent country, Grand Bahama Island is in the country of the Bahamas.
Hints:\n"""+graph2+"""
Rethink: Rethink it in the context of hints and previous response. Look for hints that explicitly mention a country in relation to Grand Bahama. The hint 'Grand Bahama' mentions it is 'containedby' the Bahamas. Therefore, the answer is "Bahamas".
### Output: ['Bahamas']

Input: where are the nfl redskins from?
CoT: Let's think step by step. First, let's identify which team the term "Redskins" refers to. The "Redskins" is the former name of an NFL team. In 2020, the team announced it would be retiring the "Redskins" name and logo after years of controversy and criticism that the name was a racial slur against Native Americans. Therefore, the team originally known as the "Redskins" is from Native Americans.
Hints:\n"""+graph3+"""
Rethink: Rethink it in the context of hints and previous response. The hints provided reference the "Washington Redskins" as the team in question. The hints also mention that the "Washington Redskins" are associated with the location "Washington, D.C.". So, I apologize for my previous incorrect answer, and the answer is Washington, D.C.
### Output: ['Washington, D.C.']\n
"""
                    
                    #query = task_description + icl +'Input: ' + query + '?\nCoT: '+cot+'\nHints:\n'+link+'\nRethink: Rethink it in the context of hints and previous response.'
                    query = task_description + icl +'Input: ' + query +'?\nHints:\n'+link+ '\nCoT: '+cot+'\n### Output: '
            else:
                
                '''
                icl = """Here are some examples:
Input: what is the name of justin bieber brother?
CoT: Let's think step by step. Justin Bieber has a half-brother from his father Jeremy Bieber's side. His name is Jaxon Bieber. Jaxon is significantly younger than Justin and has occasionally appeared with his brother at events and on Justin's social media channels. Thus, Jaxon Bieber is Justin Bieber's brother.
### Output: ['Jaxon Bieber']

Input: what country is the grand bahama island in?
CoT: Let's think step by step. Step 1: Identify the region referred to by "Grand Bahama Island." Grand Bahama Island is one of the islands in the archipelago known as the Bahamas. Step 2: Determine the political status of the Bahamas. The Bahamas is an independent country comprised of numerous islands and islets. Step 3: Conclude which country Grand Bahama Island is a part of. Since Grand Bahama Island is one of the islands in the Bahamas archipelago and the Bahamas is an independent country, Grand Bahama Island is in the country of the Bahamas.
### Output: ['Bahamas']

Input: where are the nfl redskins from?
CoT: Let's think step by step. First, let's identify which team the term "Redskins" refers to. The "Redskins" is the former name of an NFL team based in Washington D.C. We need to acknowledge that the team's name has changed. In 2020, the team announced it would be retiring the "Redskins" name and logo after years of controversy and criticism that the name was a racial slur against Native Americans. The team is now known as the Washington Commanders after undergoing a rebranding process. Therefore, the team originally known as the "Redskins" is from Washington D.C.
### Output: ['Washington, D.C.']\n
"""
                '''

                icl = """Here are some examples:
Input: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
CoT: Let's think step by step.\n1. Identify the key components of the question. The question asks for the state in which a university is located. The university is represented in sports by the "George Washington Colonials men's basketball" team.\n2. Use the name of the sports team to determine the university. The team's name is "George Washington Colonials," which suggests that the university is named after George Washington.\n3. Determine the university. The university represented by the George Washington Colonials men's basketball team is George Washington University.\n4. Find out where George Washington University is located. George Washington University is located in Washington, D.C.
Answer the question based on the information.
### Output: ['Washington, D.C.']

Input: What year did the team with Baltimore Fight Song win the Superbowl?
CoT: Let's think step by step.\n1. Identify the team associated with the Baltimore Fight Song: The Baltimore Ravens.\n2. Determine the years the Baltimore Ravens won the Super Bowl: The Ravens won the Super Bowl in the 2000 and 2012 seasons.\n3. List the years corresponding to those seasons: The 2000 season's Super Bowl was in 2001 (Super Bowl XXXV), and the 2012 season's Super Bowl was in 2013 (Super Bowl XLVII).
### Output: ['Super Bowl XLVII', 'Super Bowl XXXV']

Input: Which governor of Georgia in 2011 is the politician whose tenure started last?
CoT: Let's think step by step.\n1. Identify the timeframe: The year 2011.\n2. Clarify the role: We are looking for the governor of Georgia.\nDetermine the tenure start date: We need to find out which governor's tenure started most recently as of 2011.\nWith these steps in mind, let's proceed with the thought process: In 2011, the governor of Georgia was Nathan Deal. Nathan Deal's tenure started on January 10, 2011. Therefore, the answer is: Nathan Deal
### Output: ['Nathan Deal']\n
"""
                query = task_description + icl +'Input: ' + query + '?\nCoT: Let\'s think step by step.'
        if self.args.prompt == 'icl':
            template = 'Input: {}?\n### Output: {}\n' # ###
            icl=''
            if self.args.selecting_strategy == 'random':
                for i in range(int(self.args.k)):
                    icl += template.format(self.train_dataset[i]['question'], str(self.train_dataset[i]['answer']))
            
            if self.args.selecting_strategy == 'topk':
                query_embedding = self.get_embedding(query)
                embedding_bank = self.embedding_bank.to('cuda')
                sim = torch.cosine_similarity(embedding_bank, query_embedding, dim=-1)
                _, indices = torch.topk(sim, self.args.k, largest=True)
                indices = indices.tolist()
                indices.reverse()
                torch.cuda.empty_cache()
                for i in indices:
                    answer = self.train_dataset[i]['answer']
                    if len(answer)>20:
                        answer = self.train_dataset[i]['answer'][:20]
                    answer = str(answer)
                    icl += template.format(self.train_dataset[i]['question'], answer)
                
            if link:
                link = postprocess(link)
                query = task_description + 'Here are some examples:\n'+icl+'\nInput: ' + query + '?\nHints: '+link+'\n### Output:'
                
            else:
                query = task_description + 'Here are some examples:\n'+icl+'\nInput: ' + query + '?\n### Output:'  # ###
            
            #query = task_description + icl+'Question: ' + query + '?\nAnswer: ###'  # ###
        if self.args.prompt == 'zero_shot':
            if link:
                link = postprocess(link)
                query = task_description + 'Input: ' + query + '?\nHints:\n'+link+'\n### Output:'
            else:
                query = task_description + 'Input: ' + query + '?\n### Output:'
        
        return query

    def get_response(self, query, sample=False):
        
        #prompt_len = len(query)
        #inputs = self.tokenizer(query, return_tensors="pt")
        #input_ids = inputs["input_ids"].to(self.model.device) #.to(self.device)
        #attention_mask = inputs["attention_mask"].to(self.model.device)
        with torch.no_grad():
            '''
            outputs = self.model.generate(input_ids=input_ids,max_new_tokens=512,num_beams=5)[0]
            #outputs = self.model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=512,do_sample=True,temperature=0.8,top_p=0.95,num_return_sequences=10)
            
            response = []
            for i in outputs:
                output = self.tokenizer.decode(i, skip_special_tokens=True)[prompt_len:]#.lstrip(query)
                response.append(output)
            '''
            
            if sample:
                sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512, n=10)
                outputs = self.model.generate(query, sampling_params)
                response = []
                for i in outputs[0].outputs:
                    response.append(i.text)
                #response = outputs[0].outputs[4].text
                #print(response)
                #exit()
            else:
                sampling_params = SamplingParams(max_tokens=512, use_beam_search=True, n=5, temperature=0)
                outputs = self.model.generate(query, sampling_params)
                response = outputs[0].outputs[0].text
            
            
        #print(response)
        #exit()
        return response

    def load_graph(self):
        question_lst=[]
        paths = []
        with open('/ossfs/workspace/webcwq_retrievekg_graphtune_qatune_cot_tune_neighbor_top50_test.jsonl', 'r') as json_file:
            fcc_data = json.load(json_file)
            for line in fcc_data:
                qes = line['question']
                graph = line['graph'][:20]
                if self.args.graph_with_json:
                    graph = triple2json(graph)
                elif self.args.graph_with_yaml:
                    graph = triple2yaml(graph)
                else:
                    graph = ', '.join(graph)
                paths.append(graph)
                '''
                path = fcc_data['paths']
                if qes not in question_lst:
                    question_lst.append(qes)
                    if len(path)==0:
                        paths.append([])
                    else:
                        paths.append(path[0])
                '''
            #keys = list(fcc_data.keys())
        
        return paths

    def get_search_query(self, query, entity, graph):
        task_description = 'I will give you a question, an entity and a graph, Please identify the 5 entity pairs from the graph that are most valuable in answering this question.\n\n'
        query = task_description + 'Question: ' + query + '?\nEntity: ' + entity+'\nGraph: '+graph+'\nAnswer:'
        return query

    def search_entity(self, sample, graph=None):
        query = sample["question"]
        q_entity = sample["q_entity"]
        '''
        link = []
        for i in q_entity:
            if i in graph.keys():
                link += graph[i]
        '''
        query = self.get_search_query(query, str(q_entity), str(link))
        response = self.get_response(query)
        
        return response

    def inference(self, query, link=None, cot=None, sample_decode=False):
        #query = "Input: what is the name of justin bieber brother?\nCoT: Let's think step by step."
        #response = self.get_response(query)
        #print(response)
        #exit()
        query = query["question"]
        #q_entity = sample["q_entity"]
        #a_entity = sample['a_entity']
        
        query = self.get_final_query(query, link=link, cot=cot)
        response = self.get_response(query, sample=sample_decode)
        
        if self.args.prompt == 'icl':
            response = response.strip().strip('#').strip('\n').split('\n')[0]
        if self.args.prompt == 'cot':
            if sample_decode:
                if not self.args.only_generate_cot:
                    answer = {}
                    for i in response:
                        output = i.strip().strip('#').strip('\n').split('\nInput')[0].strip('\n')
                        #print('Prediction: '+output+'\n')
                        pattern = r'[#]+\s*(Output:)?\s*(.*)'
                        #pattern = r'["\'](.*?)["\']'
                        match = re.search(pattern, output)
                        if match:
                            output = match.group(2).strip('[').strip(']').split(',')
                            for p in output:
                                p=p.strip().strip('\'').strip('\"')
                                
                                if p in answer:
                                    answer[p] += 1
                                else:
                                    answer[p] = 1
                    response = str([key for key, value in answer.items() if value > 1])
                else:
                    #"""
                    cot=[]
                    for i in response:
                        cot.append(i.strip().strip('#').strip('\n').split('##')[0].strip('#').strip('\n'))

                    response = sorted(cot, key=lambda x:len(x))[-1]
                    #"""
                    #response = response[0].strip().strip('#').strip('\n').split('##')[0].strip('#').strip('\n')
                
                
            else:
                response = response.strip().strip('#').strip('\n').split('\nInput')[0].strip('\n')
                
                if not self.args.only_generate_cot:
                    #response = response.split('\n#')[0].strip('\n')
                    #query = query+response+'\n### Output:'
                    #response = self.get_response(query, sample=False)
                    #response = response.strip().strip('#').strip('\n').split('\n')[0]
                    ### kgtune ###
                
                    pattern = r'[#]+\s*(Output:)?\s*(.*)'
                    match = re.search(pattern, response)
                    if match:
                        response = match.group(2)
                else:
                    
                    response = response.strip().strip('#').strip('\n').split('##')[0].strip('#').strip('\n')
               
        
        return response

    def run(self):
        prediction = []
        if self.args.use_graph:
            graph = self.load_graph()
        else:
            graph = None
        if self.args.rethink:
            with open('/ossfs/workspace/webqsp_yamlkgtune_llama2_7b_chat_bgetune_cot_output.json', 'r') as file:
                cot_lst = json.load(file)
        if self.case_study:
            idx_lst = [3458, 1577, 3104, 1722, 165, 1060, 2094, 1990, 1658, 3210, 3399, 1242, 1952, 1466, 2389, 894, 2067, 570, 1154, 572, 3095, 388, 2532, 3274, 1026, 2181, 2888, 3318, 2465, 601, 1270, 404, 2989, 302, 3483, 2801, 1352, 1933, 2292, 412, 1449, 1778, 1295, 2502, 2623, 837, 2263, 1953, 1813, 2135, 1066, 255, 3297, 2247, 57, 382, 2947, 3441, 1633, 2909, 3378, 3215, 2736, 2561, 4, 2506, 2021, 3391, 1364, 999, 2991, 1332, 2882, 257, 782, 2324, 908, 977, 3290, 583, 3289, 2224, 1834, 373, 329, 1310, 2080, 2004, 446, 1234, 2257, 1192, 2894, 511, 2242, 1362, 3336, 2213, 832, 2470] #random.sample(range(len(self.test_dataset)), k=100)#[3458, 1577, 3104, 1722, 165, 1060, 2094, 1990, 1658, 3210]
            
            for idx in idx_lst:
                print('Question: '+self.test_dataset[idx]['question'])
                print('Answer: '+str(self.test_dataset[idx]['answer']))
                #entity_pair = self.search_entity(self.test_dataset[idx], graph=graph)
                if graph:
                    link = graph[idx]
                else:
                    link = None
                if self.args.rethink:
                    cot = cot_lst[idx]['cot']
                else:
                    cot=None
                
                if self.args.only_generate_cot:
                    pre = self.inference(self.test_dataset[idx], link=link,cot=cot, sample_decode=True)
                    dict = {'question': self.test_dataset[idx]['question'], 'cot': pre}
                    #print(dict)
                    #exit()
                    prediction.append(dict)
                else:
                    pre = self.inference(self.test_dataset[idx], link=link,cot=cot, sample_decode=True)
                    prediction.append(pre)
                print('###############################################\n')
            exit()

        else:
            for idx in range(12000,len(self.test_dataset)):
            #for idx in range(len(self.test_dataset)):
                #entity_pair = self.search_entity(self.test_dataset[idx], graph=graph)
                if graph:
                    link = graph[idx]
                    #print(link)
                    #random.shuffle(link)
                    #if len(link)>1:
                    #    link = link[:-1]
                else:
                    link = None
                if self.args.rethink:
                    cot = cot_lst[idx]['cot']
                else:
                    cot=None
                
                if self.args.only_generate_cot:
                    pre = self.inference(self.test_dataset[idx], link=link,cot=cot, sample_decode=True)
                    dict = {'question': self.test_dataset[idx]['question'], 'cot': pre}
                    #print(dict)
                    #exit()
                    prediction.append(dict)
                else:
                    pre = self.inference(self.test_dataset[idx], link=link,cot=cot, sample_decode=True)
                    prediction.append(pre)
                if idx % 200 == 0:
                    print("inference {} of {} samples...".format(idx, len(self.test_dataset)))

        if self.args.only_generate_cot:
            with open(self.args.save_dir,'w') as json_write:
                json.dump(prediction, json_write, indent=4)
        else:
            with open(self.args.save_dir, 'w') as json_file:
                for i in prediction:
                    json.dump(i, json_file)
                    json_file.write('\n')
                
        return prediction
