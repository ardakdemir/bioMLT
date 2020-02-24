import json
import os
from collections import defaultdict
bioasq_json_path = "/Users/ardaakdemir/bioMLT/BioASQ-SampleDataB.json"
def bioasq_to_squad(bioasq_json_path,type = 'test'):
    with open(bioasq_json_path,"r",encoding='utf-8') as b_j:
        bioasq_json = json.load(b_j)
    version = os.path.split(bioasq_json_path)[-1].split(".")[0]
    title = version
    
    qs = bioasq_json["questions"]
    all_questions = defaultdict(list)
    q_count = 0
    snip_count = 0
    not_found_count = 0
    found_count = 0
    for q in qs:
        q_count+=1
        question = q["body"]
        
        q_type = q["type"]
        if type == "train":
            ideal_answer = q['ideal_answer']
            if q_type !="summary": 
                exact_answer = q['exact_answer']
        q_id = q["id"]
        for i,snippet in enumerate(q["snippets"]):
            context = snippet['text']
            qas = {"qas":[{"id":q_id+"_{}".format(str(i+1).zfill(3)),"question":question}],
                   "context": context}
            if type == "train":
                if q_type =="yesno":
                    qas["qas"][0]["answers"] = "yes"
                    all_questions[q_type].append(qas)
                elif q_type=="factoid":
                    answers = exact_answer
                    for ans in answers:
                        ind = context.find(ans)
                        if ind !=-1:
                            qas["qas"][0]["answers"] = [{"text":ans,"answer_start":ind}]
                            all_questions[q_type].append(qas)
                            found_count+=1
                        else:
                            #print('could not find {} in {}'.format(ans,context))
                            not_found_count +=1
                            continue
                elif q_type=="list":
                    answers = exact_answer
                    for ans_list in answers:
                        for ans in ans_list:
                            ind = context.find(ans)
                            if ind !=-1:
                                qas["qas"][0]["answers"] = [{"text":ans,"answer_start":ind}]
                                all_questions[q_type].append(qas)
                                found_count+=1
                            else:
                                #print('could not find "{}" in "{}""'.format(ans,context))
                                not_found_count +=1
                                continue
                else:
                    qas["qas"][0]["answers"] = [ideal_answer]
                    all_questions[q_type].append(qas)
            else:
                all_questions[q_type].append(qas)
            snip_count+=1
    print("{} questions and {} snippets".format(q_count,snip_count))
    print("{} answers cannot found inside the snippet\n{} answers were found".format(not_found_count,found_count))
    all_jsons = {}
    for q in all_questions:
        all_jsons[q] = {"version":version,"data":[{"paragraphs":all_questions[q],"title":title}]}
    for qtype in all_jsons:
        with open("{}_squadformat_{}_{}.json".format(version,type,qtype),"w") as m_j:
            json.dump(all_jsons[qtype],m_j)
    return all_jsons


if __name__=="__main__":
    args = sys.argv
    bioasq_path = args[1]
    d_type = args[2]
    all_jsons = bioasq_to_squad(bioasq_path,type=d_type)

