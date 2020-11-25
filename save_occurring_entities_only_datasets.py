import os
import subprocess
import json
from collections import defaultdict, Counter
import sys


def read_dataset(dataset):
    dataset = [" ".join([tok.split()[0] for tok in sent.split("\n")]) for sent in
               open(dataset).read().split("\n\n")[:-1]]
    return dataset


def search_dataset(entity, dataset, first=True):
    occurences = []
    for sentence in dataset:
        if " " + entity + " " in " " + sentence + " ":
            if first:
                return [sentence]
            else:
                occurences.append(sentence)
    return occurences


def find_entity_occurences(entity_list, dataset, first=True):
    """
        This script checks for whole dataset for inclusion of entities (not just the entities)
    """
    not_found_list = []
    found_entities = defaultdict(list)
    for entity in entity_list:
        found_sentences = search_dataset(entity, dataset, first=first)
        if len(found_sentences) == 0:
            not_found_list.append(entity)
        else:
            found_entities[entity].extend(found_sentences)
    return not_found_list, found_entities


def get_entities_from_tsv_dataset(file_path, tag_type="BIO"):
    sentences = open(file_path).read().split("\n\n")[:-1]
    entities = defaultdict(Counter)
    if tag_type == "BIO":
        for sent in sentences:
            prev_tag = "O"
            curr_entity = ""
            for token in sent.split("\n"):
                if len(token.split()) > 1:
                    word = token.split()[0]
                    label = token.split()[-1]
                else:
                    word = ""
                    label = "O"
                if label != "O":
                    if label[0] == "I":
                        curr_entity = curr_entity + " " + word
                    elif label[0] == "B":
                        if prev_tag != "O":
                            if len(curr_entity) > 0:
                                entities[prev_tag][curr_entity] += 1
                        prev_tag = label[2:]
                        curr_entity = word
                else:
                    if prev_tag != "O":
                        entities[prev_tag][curr_entity] += 1
                    prev_tag = "O"
                    curr_entity = ""
            if len(curr_entity) > 0 and prev_tag != "O":
                entities[prev_tag][curr_entity] += 1
    return entities


def get_sentence_entities(sentence, labels):
    entities = []
    prev_label = "O"
    curr_entity = ""
    for s, l in zip(sentence, labels):
        if l == "O":
            if len(curr_entity) > 0:
                entities.append(curr_entity)
                curr_entity = ""
                prev_label = "O"
            continue
        if l.startswith("B-"):
            if len(curr_entity) > 0:
                entities.append(curr_entity)
                curr_entity = ""
                prev_label = l
            curr_entity = s
        else:
            curr_entity = curr_entity + " " + s
            prev_label = l
    if len(curr_entity) > 0:
        entities.append(curr_entity)
    return entities


# Read QAS dataset examples!!

def get_qas_dataset_sentences(qas_file_path):
    """
        Gets the qas dataset as Question-body, answer texts, snippets
        Used
    """
    qas_dataset = json.load(open(qas_file_path, "rb"))
    dataset = []
    for q in qas_dataset["questions"]:
        if q["type"] == "summary":
            continue
        body = q["body"]
        if q["type"] == "yesno":
            answer = ""
        elif q["type"] == "factoid":
            answer = " ".join(q["exact_answer"])
        else:
            answer = " ".join([" ".join([a for a in answer]) for answer in q["exact_answer"]])
        snippets = " ".join([x["text"] for x in q["snippets"]])
        text = " ".join([body, answer, snippets])
        dataset.append(text)
    return dataset


def save_occurring_entities_only_datasets(dataset_root_folder, qas_file_path, save_root):
    """
        Given a set of biomedical ner datasets save only sentences
        that contain entities occurring inside the qas.

        My aim is to show that when I generate subsets from these pruned datasets performance might improve??
    """
    ner_datasets = os.listdir(dataset_root_folder)
    qas_dataset = get_qas_dataset_sentences(qas_file_path)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for ner_d in ner_datasets:
        p = os.path.join(dataset_root_folder, ner_d)
        if not os.path.isdir(p):
            continue

        ner_file_path = os.path.join(p, "ent_train.tsv")
        ner_dev_file_path = os.path.join(p, "ent_devel.tsv")
        ner_test_file_path = os.path.join(p, "ent_test.tsv")

        ner_save_path = os.path.join(save_root, ner_d, "ent_train.tsv")
        ner_dev_save_path = os.path.join(save_root, ner_d, "ent_devel.tsv")
        ner_test_save_path = os.path.join(save_root, ner_d, "ent_test.tsv")
        if not os.path.exists(os.path.join(save_root, ner_d)):
            os.makedirs(os.path.join(save_root, ner_d))
        ner_entities_per_type = get_entities_from_tsv_dataset(ner_file_path, tag_type="BIO")
        ner_entities = []
        for ent_type, entities in ner_entities_per_type.items():
            ner_entities.extend(list(entities.keys()))
        print("Found {} unique entities for {}".format(len(ner_entities), ner_d))
        not_found, found_entities = find_entity_occurences(ner_entities, qas_dataset)
        found_entities = list(found_entities.keys())
        print("Found {}/{} entities inside the qas dataset".format(len(found_entities), len(ner_entities)))
        print("Found entities : {}".format(found_entities))
        print("Saving pruned dataset to {}".format(ner_save_path))
        initial_ner_dataset = open(ner_file_path, "r").read().split("\n\n")
        pruned_dataset = []
        problem_count = 0
        for example in initial_ner_dataset:
            if "\t" in example:
                sentence = [x.split("\t")[0] for x in example.split("\n")]
                labels = [x.split("\t")[-1] for x in example.split("\n")]
            elif " " in example:
                sentence = [x.split(" ")[0] for x in example.split("\n")]
                labels = [x.split(" ")[-1] for x in example.split("\n")]
            else:
                problem_count = problem_count + 1
                continue
            sentence_entities = get_sentence_entities(sentence, labels)
            if any([entity in sentence_entities for entity in found_entities]):
                pruned_dataset.append(example)
        print("Skipped {} sentences".format(problem_count))
        print(
            "{} sentences containing at least one of {} qas entities.".format(len(pruned_dataset), len(found_entities)))
        pruned_dataset = "\n\n".join(pruned_dataset)
        with open(ner_save_path, "w") as o:
            o.write(pruned_dataset)
        cmd = "cp {} {} ".format(ner_dev_file_path, ner_dev_save_path)
        subprocess.call(cmd, shell=True)
        cmd = "cp {} {} ".format(ner_test_file_path, ner_test_save_path)
        subprocess.call(cmd, shell=True)


def main():
    args = sys.argv
    dataset_root_folder, qas_file_path, save_root = args[1], args[2], args[3]
    save_occurring_entities_only_datasets(dataset_root_folder, qas_file_path, save_root)


if __name__ == "__main__":
    main()
