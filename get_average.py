import numpy as np
import sys
import os 
args = sys.argv
pref = args[1]

x = []
l = os.listdir(".")
for i in range(1,6):
    
    f = "{}{}.txt".format(pref,i)
    if not f in l:
        continue
    x.append([float(a) for a in open(f).read().strip().split(" ")])

print(np.array(x).shape)
field_names = "YesNo_Acc Factoid_Strict_Acc Factoid_Lenient_Acc Factoid_MRR List_Prec List_Rec List_F1 YesNo_macroF1 YesNo_F1_yes YesNo_F1_no".split()
res= np.mean(np.array(x),axis=0)
for f, r in zip(field_names,res):
    print("{}\t{}".format(f,r))
