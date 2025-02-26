import csv
import os

import hickle as hkl
import numpy as np
import pandas as pd


def dataload():
    Drug_feature_file = "../nci_data/drug_graph_feat"
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        feat_mat, adj_list, degree_list = hkl.load("%s/%s" % (Drug_feature_file, each))
        drug_feature[each.split(".")[0]] = [feat_mat, adj_list, degree_list]

    exp = pd.read_csv("../nci_data/nci60_gene_exp.csv", index_col=0).T

    mutation = pd.read_csv("../nci_data/nci60GeneMut.csv", index_col=0).T
    mutation.index = exp.index

    methylation = pd.read_csv(
        "../nci_data/nci60_methylation.csv", index_col=0
    ).T.fillna(0)

    nb_celllines = exp.shape[0]
    nb_drugs = len(drug_feature)

    return (drug_feature, exp, mutation, methylation, nb_celllines, nb_drugs)
