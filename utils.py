import os
dicts = {"ner":  {"token" : 0 , "truth" : 1 , "ner_pred " : 2}} 
def sort_dataset(dataset,desc=True, sort = True):

    idx = [i for i in range(len(dataset))]
    if not sort:
        return dataset, idx
    zipped = list(zip(dataset,idx))
    zipped.sort(key = lambda x : len(x[0]))
    if desc:
        zipped.reverse()
    dataset, orig_idx = list(zip(*zipped))
    return dataset, orig_idx

def conll_writer(file_name, content, field_names, task_name,verbose=False):
    out = open(file_name,'w',encoding = 'utf-8')
    task_dict = dicts[task_name]
    if verbose:
        out.write("{}\n".format("\t".join([k for k in task_dict])))
    init = ["-" for i in range(len(task_dict))]
    for sent in content:
        for id,tok in enumerate(sent):
            for i,f in enumerate(field_names):
                init[task_dict[f]] = tok[i]
                if type(tok[i])==int:
                    init[task_dict[f]]=str(tok[i])
            if task_name == 'dep':
                init[0] =  str(id+1)
            out.write("{}\n".format("\t".join(init)))
        out.write("\n")
    out.close()
def unsort_dataset(dataset,orig_idx):
    zipped = list(zip(dataset,orig_idx))
    zipped.sort(key = lambda x : x[1])
    dataset , _ = list(zip(*(zipped)))
    return dataset
