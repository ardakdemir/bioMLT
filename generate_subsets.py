import os
import numpy as np
import subprocess
import sys

file_names = ["ent_train.tsv", "ent_devel.tsv", "ent_test.tsv"]


def get_small_dataset(file, shrink=None, size=None):
    limit = size
    sentences = open(file).read().split("\n\n")[:-1]
    if shrink is not None:
        limit = int(len(sentences) / shrink)
    if shrink is None and size is None:
        limit = len(sentences)
    np.random.shuffle(sentences)
    x = sentences[:limit]
    return "\n\n".join([sent for sent in x])


def generate_small_datasets(folder, save_folder, shrink=None, size=None):
    datasets = os.listdir(folder)
    for dataset in datasets:
        path = os.path.join(folder, dataset)
        if not os.path.isdir(path) or any([f not in os.listdir(path) for f in file_names]):
            print("Skipping {} ".format(dataset))
        dataset_save_folder = os.path.join(save_folder, dataset)
        if not os.path.exists(dataset_save_folder):
            os.makedirs(dataset_save_folder)
        for file in file_names:
            file_path = os.path.join(path, file)
            save_path = os.path.join(dataset_save_folder, file)
            if file == "ent_test.tsv":
                cmd = "cp {} {}".format(file_path, save_path)
                subprocess.call(cmd, shell=True)
            else:
                shrinked = get_small_dataset(file_path, shrink=shrink, size=size)
                with open(save_path, "w") as s_p:
                    s_p.write(shrinked)



def main():
    args = sys.argv
    root_folder, save_folder_prefix, repeat = args[1], args[2], args[3]
    subset_sizes = [1000,2000,5000,10000,20000]
    for size in subset_sizes:
        for r in range(repeat):
            save_name = save_folder_prefix + "{}_".format(str(size) if size is not None else "All")
            save_name = save_name + str(r)
            generate_small_datasets(root_folder,save_name,size = size)
            if size is None:
                break

if __name__ == "__main__":
    main()
