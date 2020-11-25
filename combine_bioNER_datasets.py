import os
import time
import sys
from collections import defaultdict, Counter
import subprocess

def read_datasets_with_labels_from_folder(data_folder, file_name="ent_train.tsv"):
    dataset_names = os.listdir(data_folder)
    datasets = {}
    for d in dataset_names:
        p = os.path.join(data_folder, d)
        if not os.path.isdir(p):
            print("Skipping {}".format(d))
            continue
        p = os.path.join(p, file_name)
        initial_ner_dataset, sentences, labels = read_dataset_with_labels(p)
        datasets[d] = {"examples": initial_ner_dataset, "sentences": sentences, "labels": labels}
    return datasets


def get_diff_ratio(entity_set_1, entity_set_2, same_tag=True):
    """
        Currently only counting the occurrence and non-occurrence
        Used for finding the entity-level (not word-level) rate of OOVs
        If same_tag, only checks the inclusion under the same tag. Used for datasets with different label sets
    """
    total = 0
    diff = 0
    not_found_entities = set()
    found_entities = set()
    for entity in entity_set_1.keys():
        for word in entity_set_1[entity]:
            if same_tag:
                if word not in entity_set_2[entity]:
                    diff += entity_set_1[entity][word]
                    not_found_entities.add(word)
                else:
                    found_entities.add(word)
            else:
                if not any([word in entity_set_2[ent] for ent in entity_set_2.keys()]):
                    diff += entity_set_1[entity][word]
                    not_found_entities.add(word)
                else:
                    found_entities.add(word)
            total += entity_set_1[entity][word]
    return diff / total, list(not_found_entities), list(found_entities)


def read_dataset_with_labels(dataset_file_path):
    initial_ner_dataset = open(dataset_file_path, "r").read().split("\n\n")
    sentences = []
    labels = []
    for example in initial_ner_dataset:
        if "\t" in example:
            sentence = [x.split("\t")[0] for x in example.split("\n")]
            label = [x.split("\t")[-1] for x in example.split("\n")]
        elif " " in example:
            sentence = [x.split(" ")[0] for x in example.split("\n")]
            label = [x.split(" ")[-1] for x in example.split("\n")]
        else:
            problem_count = problem_count + 1
            continue
        sentences.append(sentence)
        labels.append(label)
    return initial_ner_dataset, sentences, labels


def get_entities_from_tsv_dataset(file_path, tag_type="BIO"):
    sentences = open(file_path).read().split("\n\n")[:-1]
    entities = defaultdict(Counter)
    if tag_type == "BIO":
        for sent in sentences:
            prev_tag = "O"
            curr_entity = ""
            for token in sent.split("\n"):
                word = token.split()[0]
                label = token.split()[-1]
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


def convert_to_tsv_example(sentence, labels):
    return "\n".join(["\t".join([s, l]) for s, l in zip(sentence, labels)])


def annotate_sentence(entity_dict, sentence, labels):
    """
        Given a dict of entities update the labels of
    """
    my_sentence = " ".join(sentence)
    count = 0
    entity_annotation_counts = defaultdict(int)

    for entity, lab in entity_dict.items():
        if " " + entity + " " in " " + my_sentence + " ":
            words = entity.split(" ")
            word_one = words[0]
            index = sentence.index(word_one)
            if all([label == "O" for label in labels[index:index + len(words)]]):
                labels[index] = "B-" + lab
                for x in range(index + 1, index + len(words)):
                    labels[x] = "I-" + lab
                entity_annotation_counts[entity] += 1
                count = count + 1
    return sentence, labels, count, entity_annotation_counts


def get_entities_from_folder(data_folder, train_file_name, test_file_name):
    oov_rates = {}
    datasets = os.listdir(data_folder)
    all_train_entities = {}
    all_test_entities = {}
    for d in datasets:
        folder = os.path.join(data_folder, d)
        if not os.path.isdir(folder) or train_file_name not in os.listdir(folder):
            continue
        entity_type = d
        test_data_path = "{}/{}".format(folder, test_file_name)
        test_entities = get_entities_from_tsv_dataset(test_data_path)
        train_data_path = "{}/{}".format(folder, train_file_name)
        train_entities = get_entities_from_tsv_dataset(train_data_path)
        oov_rate, oovs, found_entities = get_diff_ratio(test_entities, train_entities)
        oov_rates[entity_type] = oov_rate
        all_train_entities[entity_type] = train_entities
        all_test_entities[entity_type] = test_entities
    return all_train_entities, all_test_entities, oov_rates


# When combining datasets, annotate all sentences with entities of other datasets
def combine_datasets(dataset_root_folder, save_folder_name, min_count=3):
    """

        Given a set of entity datasets we concatenate them and annotate all entities that occur in other datasets.
        Currently, we do not consider the dev and test datasets. They are just concatenated...

    :param dataset_root_folder: Folder containing all bioNER datasets
    :param save_folder_name: folder name to store the combined datasets


    """
    save_folder = os.path.join(dataset_root_folder,save_folder_name)
    if os.path.exists(save_folder):
        cmd = "rm -r {}".format(save_folder)
        subprocess.call(cmd,shell=True)

    ner_dataset_names = os.listdir(dataset_root_folder)
    combined_dataset = []
    b = time.time()
    all_train_entities, all_test_entities, oov_rates = get_entities_from_folder(dataset_root_folder, "ent_train.tsv",
                                                                                "ent_test.tsv")
    all_entities = set()
    entity_labels = {}
    entity_counts = {}
    min_occurrence = min_count
    not_consider_count = 0
    for dataset_name, train_entities in all_train_entities.items():
        for k, v in train_entities.items():
            for ent in v:
                if v[ent] > min_occurrence:
                    entity_labels[ent] = k
                    entity_counts[ent] = v[ent]
                else:
                    not_consider_count = not_consider_count + 1
    print("{} entities in total. {} entities ignored.".format(len(entity_labels), not_consider_count))
    entity_annotation_counts = defaultdict(int)
    e = time.time()
    t = round(e - b, 3)
    print("Got entities in {} seconds".format(t))
    annotated_sentence_count = 0
    b = time.time()
    datasets = read_datasets_with_labels_from_folder(dataset_root_folder, file_name="ent_train.tsv")
    e = time.time()
    t = round(e - b, 3)
    print("Got datasets with labels in {} seconds".format(t))
    annotation_count = 0
    total_sentence_num = sum([len(v["examples"]) for k, v in datasets.items()])
    print("{} sentences in total".format(total_sentence_num))
    b = time.time()
    for name, dataset in datasets.items():
        examples, sentences, labels = dataset["examples"], dataset["sentences"], dataset["labels"]
        for example, sentence, label in zip(examples, sentences, labels):
            new_s, new_l, my_count,my_entity_annotation_counts = annotate_sentence(entity_labels, sentence, label)
            if my_count > 0:
                annotation_count = annotation_count + my_count
                annotated_sentence_count = annotated_sentence_count + 1
                for x in my_entity_annotation_counts:
                    entity_annotation_counts[x] += my_entity_annotation_counts[x]
            annotated_example = convert_to_tsv_example(new_s, new_l)
            combined_dataset.append(annotated_example)
    e = time.time()
    t = round(e - b, 3)

    print("Annotated sentences with missing annnotations in {} seconds".format(t))
    print("{} entities are annotated in total".format(annotation_count))
    print("{} sentences inside the combined dataset in total".format(len(combined_dataset)))
    print("{} sentences are annotated with new labels.".format(annotated_sentence_count))
    print("Annotation counts: {}".format(entity_annotation_counts))
    print("{} unique entities are annotated in other datasets in total".format(len(entity_annotation_counts)))


    combined_dataset = "\n\n".join(combined_dataset)
    save_folder = os.path.join(dataset_root_folder, save_folder_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(dataset_root_folder, save_folder_name, "ent_train.tsv")
    with open(save_path, "w") as o:
        o.write(combined_dataset)

    dev_name = "ent_devel.tsv"
    test_name = "ent_test.tsv"
    ner_dev_save_path = os.path.join(dataset_root_folder, save_folder_name, "ent_devel.tsv")
    ner_test_save_path = os.path.join(dataset_root_folder, save_folder_name, "ent_test.tsv")
    with open(ner_dev_save_path, "w") as o:
        s = "\n".join(
            [open(os.path.join(dataset_root_folder, f, dev_name)).read() for f in os.listdir(dataset_root_folder)])
        o.write(s)
    with open(ner_test_save_path, "w") as o:
        s = "\n".join(
            [open(os.path.join(dataset_root_folder, f, test_name)).read() for f in os.listdir(dataset_root_folder)])
        o.write(s)


def main():
    args = sys.argv
    dataset_root_folder, save_folder_name, min_count = args[1], args[2], int(args[3])
    combine_datasets(dataset_root_folder, save_folder_name, min_count=min_count)


if __name__ == "__main__":
    main()
