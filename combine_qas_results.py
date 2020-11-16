import os
import sys

def get_result_dict(root,folder_pref, result_file_name):
    folders = os.listdir(root)
    results = {}
    for folder in folders:
        if folder.beginswith(folder_pref):
            f_p = os.path.join(root, folder_pref)
            if os.isdir(f_p):
                dataset_name = folder.split("_")[-1]
                file_path = os.path.join(result_file_name)
                results[dataset_name] = {}
                with open(file_path, "r") as f:
                    r = f.read().split("\n")[-1].split()[1:]
                    results[dataset_name]["yesno"] = r[1]
                    results[dataset_name]["list"] = r[0]
                    results[dataset_name]["factoid"] = r[2]
    return results


if __name__ == "__main__":
    args = sys.argv
    root = args[1]
    folder_pref = args[2]
    result_file_name = args[3]
    result_dict = get_result_dict(root, folder_pref, result_file_name)
