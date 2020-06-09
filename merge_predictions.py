import json
import os
import sys

file_pref = sys.argv[1]
out_name = sys.argv[2]
def merge_jsons(file_list,out_name):
    
    final_dic ={'system' : 'dummy_name',"questions" : [] }
    for f in file_list:
        j =  json.load(open(f,'r'))
        for q in j['questions']:
            new_q = {'id': q['id'], 'exact_answer' : q['exact_answer'],'ideal_answer': q['ideal_answer']}
            final_dic['questions'].append(new_q)
    with open(out_name,'w') as o:
        json.dump(final_dic,o)

file_list = [file_pref+"_{}".format(x) for x  in ['factoid','list', 'yesno'] ]
#file_list = ['converted_allquestionspredictions_list_1', 'converted_allquestionspredictions_factoid_1']
merge_jsons(file_list,out_name)
