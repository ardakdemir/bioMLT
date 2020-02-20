import numpy as np
import sys

args = sys.argv
pref = args[1]

x = []
for i in range(1,6):
    f = "{}{}.txt".format(pref,i)
    x.append([float(a) for a in open(f).read().split(" ")])

print(np.array(x).shape)
print(np.mean(np.array(x),axis=0))
