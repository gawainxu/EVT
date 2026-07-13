import argparse
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis

from Test import AUROC

def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="odin")
    parser.add_argument("--inliers_features_path", type=str, default=".")
    parser.add_argument("--outliers_features_path", type=str, default=".")

    opt = parser.parse_args()
    return opt


def odin(inliers_features, outliers_features):

    inliers_outputs = inliers_features["output"]
    outliers_outputs = outliers_features["output"]

    inliers_outputs = [io.numpy() for io in inliers_outputs]
    outliers_outputs = [oo.numpy() for oo in outliers_outputs]
    inliers_outputs = np.array(inliers_outputs)
    outliers_outputs = np.array(outliers_outputs)

    inliers_scores = np.max(inliers_outputs, axis=1)
    outliers_scores = np.max(outliers_outputs, axis=1)

    labels_binary = np.concatenate((np.ones_like(inliers_scores), np.zeros_like(outliers_scores)))
    scores = np.concatenate((inliers_scores, outliers_scores))
    auroc = AUROC(labels_binary, scores)

    return auroc


def energy(inliers_features, outliers_features):

    inliers_linear3 = inliers_features["linear3"]
    outliers_linear3 = outliers_features["linear3"]

    inliers_linear3 = [il3.numpy() for il3 in inliers_linear3]
    outliers_linear3 = [ol3.numpy() for ol3 in outliers_linear3]
    inliers_linear3 = np.array(inliers_linear3)
    outliers_linear3 = np.array(outliers_linear3)

    inliers_scores = np.linalg.norm(inliers_linear3, axis=1)
    outliers_scores = np.linalg.norm(outliers_linear3, axis=1)

    labels_binary = np.concatenate((np.ones_like(inliers_scores), np.zeros_like(outliers_scores)))
    scores = np.concatenate((inliers_scores, outliers_scores))
    auroc = AUROC(labels_binary, scores)

    return auroc

def mahalanobis_distance(training_features, inliers_features, outliers_features, num_classes):

    inliers_linear3 = inliers_features["linear3"]
    outliers_linear3 = outliers_features["linear3"]
    inliers_labels = inliers_features["labels"]

    inliers_linear3 = [il3.numpy() for il3 in inliers_linear3]
    outliers_linear3 = [ol3.numpy() for ol3 in outliers_linear3]
    inliers_linear3 = np.array(inliers_linear3)
    outliers_linear3 = np.array(outliers_linear3)
    inliers_labels = np.array(inliers_labels)

    def sort_features(features, labels, num_classes):

        sorted_features = [[] for _ in range(num_classes)]
        for f, l in zip(features, labels):
            f = np.squeeze(f)
            sorted_features[l].append(f)

        return np.array(sorted_features)

    inliers_linear3 = sort_features(inliers_linear3)

    def feature_stats(features):

        stats = []
        for features in features:
            features = np.squeeze(np.array(features))
            mu = np.mean(features, axis=0)
            var = np.cov(features.astype(float), rowvar=False)

            stats.append((mu, var))

        return stats

    stats = feature_stats(inliers_linear3)

    dis_logits_out = []
    for features in outliers_linear3:
        diss = []
        for i, (mu, var) in enumerate(stats):
            dis = mahalanobis(features, mu, np.linalg.inv(var))
            diss.append(dis)

        dis_logits_out.append(-np.min(np.array(diss)) / np.sum(np.array(diss)))

    dis_logits_in = []
    for features in inliers_linear3:
        diss = []
        for i, (mu, var) in enumerate(stats):
            dis = mahalanobis(features, mu, np.linalg.inv(var))
            diss.append(dis)

        dis_logits_in.append(-np.min(np.array(diss)) / np.sum(np.array(diss)))

    labels_binary = np.concatenate((np.ones_like(dis_logits_in), np.zeros_like(dis_logits_out)))
    scores = np.concatenate((dis_logits_in, dis_logits_out))
    auroc = AUROC(labels_binary, scores)

    return auroc



if __name__ == "__main__":

    opt = getParse()

    with open(opt.inliers_features_path, "rb") as f:
        inliers_features = pickle.load(f)

    with open(opt.outliers_features_path, "rb") as f:
        outliers_features = pickle.load(f)

