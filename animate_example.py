import numpy as np
import matplotlib.pyplot as py
import matplotlib.animation as animation
from sklearn.decomposition import PCA
import h5py

#Get data
feats = []
with h5py.File("bert_vectors/BioASQ-training8b_factoid.hdf5","r") as h:
    feats = h["vectors"][:]
feats = feats.reshape(-1,768)
feats.shape
# %matplotlib notebook
pca = PCA(n_components=2)
pca_vectors = pca.fit_transform(feats)

fig = py.figure(2)
ax = py.axes(xlim=(-10, 10), ylim=(-10, 10))
scat = ax.scatter([], [], s=60)


def animate(i):
    scat.set_offsets(pca_vectors[:i,:])
    return scat,


# Init only required for blitting to give a clean slate.
def init():
    scat.set_offsets([])
    return scat,

ani = animation.FuncAnimation(fig, animate, init_func=init,frames=len(pca_vectors),
                               interval=10, blit=False, repeat=False)
py.show()