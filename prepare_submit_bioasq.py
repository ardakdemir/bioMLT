import json
import os
import sys
import subprocess


## adds snippets and empty ideal answers to summary type questions
## make sure ideal answers are strings not lists
def prepare_submit(sys_name, myjson, test_batch):
    my_j = json.load(open(myjson, 'r'))
    test_json = json.load(open(test_batch, 'r'))
    my_j["system"] = sys_name
    for question in test_json["questions"]:
        id = question['id']
        found = False
        snips = question["snippets"]
        for i, my_question in enumerate(my_j["questions"]):
            my_id = my_question['id']
            if my_id == id:
                found = True
                my_j["questions"][i]["ideal_answer"] = my_j["questions"][i]["ideal_answer"][0]
                my_j["questions"][i]["snippets"] = snips
        if not found:
            type = question["type"]
            if type == "summary":
                my_j["questions"].append({"id": id, "snippets": snips, "ideal_answer": 'Dummy answer'})
    save_path = "prepared_{}".format(myjson)
    json.dump(my_j, open(save_path, "w"))
    print("Saving prepared json to {}".format(save_path))
    return save_path


def compare_predictions(pred_path, test_path, write=True):
    my_json = json.load(open(pred_path, 'rb'))
    test_json = json.load(open(test_path, 'rb'))
    my_questions = my_json["questions"]
    questions = test_json["questions"]
    found_tot = 0
    for q in questions:
        c = 0
        for i, my_q in enumerate(my_questions):
            if my_q["id"] == q["id"]:
                my_json["questions"][i]["snippets"] = q["snippets"]
                if q["type"] in ["factoid", "list", "yesno"]:
                    my_json["questions"][i]["ideal_answer"] = my_json["questions"][i]["ideal_answer"][0]
                c += 1
                break
        if c == 0:
            print("ANSWER NOT FOUND FOR {} Question type {}".format(q["id"],q[type]))
            if q["type"] == "yesno":
                my_json["questions"].append({"id": q["id"],
                                             "exact_answer": "yes",
                                             "ideal_answer": "Dummy",
                                             "snippets": q["snippets"]})
            else:
                my_json["questions"].append({"id": q["id"],
                                             "ideal_answer": "Dummy answer",
                                             "snippets": q["snippets"]})
            print(q['type'])
        else:
            found_tot += 1
    print("{} out of {} have answers inside".format(found_tot, len(questions)))
    save_path = "final_{}".format(pred_path)
    print("Saving final predictions to {}".format(save_path))
    with open(save_path, "w") as o:
        json.dump(my_json, o)
    return save_path


## verifies number of questions 
def verify_submission(myjson, test_batch):
    print("Verifying {}".format(myjson))
    my_j = json.load(open(myjson, 'r'))
    test_json = json.load(open(test_batch, 'r'))
    print("My number of questions ", len(my_j["questions"]))
    print("Gold number of questions ", len(test_json["questions"]))


args = sys.argv
save_folder = args[4]
save_path = prepare_submit(args[1], args[2], args[3])
verify_submission(save_path, args[3])
final_pred_json_path = compare_predictions(save_path, args[3])

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cmd = "cp {} {}".format(final_pred_json_path, save_folder)
subprocess.call(cmd, shell=True)
print("Submission file stored inside: {}".format(save_folder))
