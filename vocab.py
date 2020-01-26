## not sure if root is needed at this stage
UNK = "[UNK]"
PAD = "[PAD]"
START_TAG = "[CLS]"
END_TAG   = "[SEP]"
PAD_IND = 0
START_IND = 1
END_IND = 2
UNK_IND = 3
VOCAB_PREF = {PAD : PAD_IND, START_TAG : START_IND, END_TAG:END_IND, UNK : UNK_IND}

class Vocab:

    def __init__(self,w2ind):
        self.w2ind =  w2ind
        self.ind2w = [x for x in w2ind.keys()]
    def __len__(self):
        return len(self.w2ind)
    def map(self,units):
        return [self.w2ind.get(x,UNK_IND) for x in units]

    def unmap(self,idx):
        return [self.ind2w[i] for i in idx]
