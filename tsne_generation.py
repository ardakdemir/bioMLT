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
    args = parser.parse_args()
    args.device = device
    return args

#Get data
def get_stored_features(file_name):
    with h5py.File(file_name,"r") as h:
        feats = h["vectors"][:]
    feats = feats.reshape(-1,768)
    return feats

def tsne_generation(vectors):
    b = time.time()
    tsne = TSNE(n_components=2, n_iter=250, n_iter_without_progress=50)
    tsne_vectors = tsne.fit_transform(vectors)
    e = time.time()
    t = round(e - b, 3)
    print("{} tSNE are generated in {} seconds".format(len(tsne_vectors),t))
    return tsne_vectors

def store_tsne_vectors():
    args = parse_args()
    ner_file_path, qas_file_path  = args.ner_vector_file, args.qas_vector_file
    limit = 5000
    save_folder = args.save_folder

    ner_file_name = os.path.split(ner_file_path)[-1].split(".")[0]
    qas_file_name = os.path.split(qas_file_path)[-1].split(".")[0]

    ner_feats = get_stored_features(ner_file_path)[:limit]
    qas_feats = get_stored_features(qas_file_path)[:limit]

    ner_length = len(ner_feats)
    qas_length = len(qas_feats)
    vectors = np.vstack([ner_feats,qas_feats])
    print("Shape of features: {}".format(vectors.shape))

    tsne_vectors = tsne_generation(vectors)
    ner_tsne = tsne_vectors[:ner_length]
    qas_tsne = tsne_vectors[ner_length:]


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ner_save_path = os.path.join(save_folder,ner_file_name+".hdf5")
    qas_save_path = os.path.join(save_folder,qas_file_name+".hdf5")
    with h5py.File(ner_save_path,"w") as h:
            h["vectors"] = np.array(ner_tsne)
    with h5py.File(qas_save_path, "w") as h:
        h["vectors"] = np.array(qas_tsne)


def main():
    store_tsne_vectors()

if __name__ == "__main__":
    main()