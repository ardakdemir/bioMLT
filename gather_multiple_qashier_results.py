import os


def get_single_result(result_folder, file_name="qas_latex_table", offset=2):
    file_path = os.path.join(result_folder, file_name)
    results = {}
    with open(file_path, "r") as r:
        f = r.read().split("\n")[offset:]
        for line in f:
            result_line = line.split("&")
            if len(result_line) < 2:
                continue
            model_name = result_line[0]
            list_f1 = float(result_line[1])
            list_exact = float(result_line[2])
            factoid_f1 = float(result_line[3])
            factoid_exact = float(result_line[4])
            yes_f1 = float(result_line[5][:-2])
            results[model_name] = {"factoid_exact": factoid_exact,
                                   "factoid_f1": factoid_f1,
                                   "list_exact": list_exact,
                                   "list_f1": list_f1,
                                   "yesno_f1": yes_f1}
    return results


def printDict(results_dict):
    title = "\t".join(["Folder","Model","Factoid Exact", "Factoid F1", "List Exact", "List F1", "YesNo F1"]) + "\n"
    table = title
    for exp_name,results in results_dict.items():
        keys = ["factoid_exact","factoid_f1","list_exact","list_f1","yesno_f1"]
        for model,result in results.items():

            row = "\t".join([exp_name,model] + [str(round(result[key],3))for key in keys]) + "\n"
            table = table + row
    return table
def get_multiple_results(root, folder_pref):
    x = os.listdir(root)
    results = {}
    for f in x:
        if f.startswith(folder_pref):
            folder_exp_name = f[len(folder_pref):]
            size, repeat = folder_exp_name.split("_")
            path = os.path.join(root, f)
            result = get_single_result(path, file_name="qas_latex_table", offset=2)
            results[folder_exp_name] = result
    return results
# results = get_single_result(".")
folder_pref = "qashiers_"
results_dict = get_multiple_results(".", folder_pref)
results_table = printDict(results_dict)
print(results_table)
# print(results)
