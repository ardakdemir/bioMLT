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

def unsort_dataset(dataset,orig_idx):
    zipped = list(zip(dataset,orig_idx))
    zipped.sort(key = lambda x : x[1])
    dataset , _ = list(zip(*(zipped)))
    return dataset
