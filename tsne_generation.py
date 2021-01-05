"""

It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data)
 to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high.
  This will suppress some noise and speed up the computation of pairwise distances between samples.
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_vector_file', type=str, default='bert_vectors/All-entities.hdf5')
    parser.add_argument('--qas_vector_file', type=str, default='bert_vectors/BioASQ-training8b.hdf5')
    parser.add_argument('--save_folder', type=str, default='tsne_vectors')
    parser.add_argument('--patience', type=int, default=100, help="Number of epochs without progress.")
    parser.add_argument('--n_iter', type=int, default=500, help="Number of epochs.")
    parser.add_argument('--tsne_dim', type=int, default=2, help="Number of dims.")
    parser.add_argument('--limit', type=int, default=500000, help="Number of vectors from each dataset.")
    parser.add_argument('--with_pca', action="store_true", default=False, help="Whether to apply pca before tsne or not (to speed up)...")
    parser.add_argument('--pca_dim', type=int, default=50, help="Number of components for pca...")

    args = parser.parse_args()
    args.device = device
    return args



# Get data
def get_stored_features(file_name):
    with h5py.File(file_name, "r") as h:
        feats = h["vectors"][:]
    feats = feats.reshape(-1, 768)
    return feats


def tsne_generation(vectors, args):
    patience = args.patience
    n_iter = args.n_iter
    tsne_dim = args.tsne_dim
    b = time.time()
    tsne = TSNE(n_components=tsne_dim, n_iter=n_iter, n_iter_without_progress=patience)
    tsne_vectors = tsne.fit_transform(vectors)
    e = time.time()
    t = round(e - b, 3)
    print("{} tSNE are generated in {} seconds".format(len(tsne_vectors), t))
    return tsne_vectors


def plot_visualization(vector_array,names,save_path):
    plt.figure()
    for vecs,name in zip(vector_array,names):
        plt.scatter(vecs[:, 0], vecs[:, 1], label=name, zorder=0)
    plt.legend()
    plt.savefig(save_path)

def store_tsne_vectors():
    args = parse_args()
    ner_file_path, qas_file_path = args.ner_vector_file, args.qas_vector_file
    limit = int(args.limit)
    save_folder = args.save_folder
    exp_name = "niter_{}_patience_{}".format(args.n_iter,args.patience)
    plot_save_path = os.path.join(save_folder,"tsne_visualization_{}.png".format(exp_name))

    ner_file_name = os.path.split(ner_file_path)[-1].split(".")[0]
    qas_file_name = os.path.split(qas_file_path)[-1].split(".")[0]

    ner_feats = get_stored_features(ner_file_path)

    qas_feats = get_stored_features(qas_file_path)
    print("Ner feats shape: {}".format(ner_feats.shape))
    print("Qas feats shape: {}".format(qas_feats.shape))

    ner_feats = ner_feats[:limit]
    qas_feats = qas_feats[:limit]

    ner_length = len(ner_feats)
    qas_length = len(qas_feats)
    vectors = np.vstack([ner_feats, qas_feats])
    print("Shape of features: {}".format(vectors.shape))

    if args.with_pca:
        print("Applying pca first to reduce dimensionality...")
        pca = PCA(n_components=args.pca_dim)
        vectors = pca.fit_transform(vectors)
        print("Output shape of PCA: {}".format(vectors.shape))
    tsne_vectors = tsne_generation(vectors, args)
    ner_tsne = tsne_vectors[:ner_length]
    qas_tsne = tsne_vectors[ner_length:]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ner_save_path = os.path.join(save_folder, ner_file_name + ".hdf5")
    qas_save_path = os.path.join(save_folder, qas_file_name + ".hdf5")
    with h5py.File(ner_save_path, "w") as h:
        h["vectors"] = np.array(ner_tsne)
    with h5py.File(qas_save_path, "w") as h:
        h["vectors"] = np.array(qas_tsne)
    plot_visualization([ner_tsne,qas_tsne],["ner","qas"],plot_save_path)

def main():
    store_tsne_vectors()


if __name__ == "__main__":
    main()
