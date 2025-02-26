import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from graphset import *
from scipy.sparse import coo_matrix


def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype="float32")
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def FeatureExtract(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat, adj_list, _ = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data


def cmask(num, ratio, seed):
    mask = np.ones(num, dtype=bool)
    mask[0 : int(ratio * num)] = False
    np.random.seed(seed)
    np.random.shuffle(mask)
    return mask


def process(
    drug_feature,
    mutation_feature,
    gexpr_feature,
    methylation_feature,
    train_df,  # Training data DataFrame
    valid_df,  # Validation data DataFrame (optional)
    test_df,   # Test data DataFrame
    nb_celllines=None,
    nb_drugs=None,
):
    # Extract data from DataFrames
    # Each DataFrame should have columns: cell_line_id, drug_id, label
    # where label is 0 or 1

    # Convert DataFrames to list format
    def df_to_list(df):
        if df is None or df.empty:
            return []
        # Use 0/1 labels
        return df[['Cell', 'Drug', 'labels']].values.tolist()

    train_data = df_to_list(train_df)
    test_data = df_to_list(test_df)
    valid_data = df_to_list(valid_df)

    # Combine all data and get unique cell lines and drug IDs
    data_new = train_data + valid_data

    cellineid = list(set([item[0] for item in data_new]))
    cellineid.sort()
    pubmedid = list(set([item[1] for item in data_new]))
    pubmedid.sort()

    # Auto-calculate nb_celllines and nb_drugs if not provided
    if nb_celllines is None:
        nb_celllines = len(cellineid)
    if nb_drugs is None:
        nb_drugs = len(pubmedid)

    cellmap = list(zip(cellineid, list(range(len(cellineid)))))
    pubmedmap = list(
        zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid))))
    )

    # Drug feature input processing
    pubid = [str(item[0]) for item in pubmedmap]
    drug_feature_df = pd.DataFrame(drug_feature).T
    drug_feature_df = drug_feature_df.loc[pubid]
    atom_shape = drug_feature_df[0].iloc[0].shape[-1]
    drug_data = FeatureExtract(drug_feature_df)

    # Cell line feature input processing
    cellid = [item[0] for item in cellmap]
    gexpr_feature_df = gexpr_feature.loc[cellid]
    mutation_feature_df = mutation_feature.loc[cellid]
    methylation_feature_df = methylation_feature.loc[cellid]

    mutation = torch.from_numpy(np.array(mutation_feature_df, dtype="float32"))
    mutation = torch.unsqueeze(mutation, dim=1)
    mutation = torch.unsqueeze(mutation, dim=1)
    gexpr = torch.from_numpy(np.array(gexpr_feature_df, dtype="float32"))
    methylation = torch.from_numpy(np.array(methylation_feature_df, dtype="float32"))

    # Prepare data loaders
    drug_set = Data.DataLoader(
        dataset=GraphDataset(graphs_dict=drug_data),
        collate_fn=collate,
        batch_size=nb_drugs,
        shuffle=False,
    )
    cellline_set = Data.DataLoader(
        dataset=Data.TensorDataset(mutation, gexpr, methylation),
        batch_size=nb_celllines,
        shuffle=False,
    )

    # Create masks from split data
    def prepare_data(data_subset):
        if not data_subset:
            return np.array([]).reshape(0, 3)

        cellline_num = np.array([
            [j[1] for j in cellmap if i[0] == j[0]][0] for i in data_subset
        ])
        pubmed_num = np.array([
            [j[1] for j in pubmedmap if i[1] == j[0]][0] for i in data_subset
        ])
        # Use 0/1 labels
        label_num = np.array([i[2] for i in data_subset])

        pairs = np.vstack((cellline_num, pubmed_num, label_num)).T
        pairs = pairs[pairs[:, 2].argsort()]
        pairs[:, 1] -= nb_celllines  # Adjust indices
        return pairs

    # Prepare each dataset
    train_pairs = prepare_data(train_data)
    test_pairs = prepare_data(test_data)
    valid_pairs = prepare_data(valid_data)

    # Create masks
    def create_mask(pairs, shape):
        if pairs.shape[0] == 0:
            return torch.zeros(shape[0] * shape[1], dtype=torch.bool)
        mask = coo_matrix(
            (np.ones(pairs.shape[0], dtype=bool), (pairs[:, 0], pairs[:, 1])),
            shape=shape
        ).toarray()
        return torch.from_numpy(mask).view(-1)

    train_mask = create_mask(train_pairs, (nb_celllines, nb_drugs))
    test_mask = create_mask(test_pairs, (nb_celllines, nb_drugs))
    valid_mask = create_mask(valid_pairs, (nb_celllines, nb_drugs))

    # ラベル行列の作成（訓練と検証データのみを使用）
    train_valid_pairs = np.vstack([train_pairs, valid_pairs])
    label_matrix = np.zeros((nb_celllines, nb_drugs))

    # 訓練と検証データのラベルを設定
    for pair in train_valid_pairs:
        cell_idx, drug_idx, label = pair
        label_matrix[int(cell_idx), int(drug_idx)] = label

    # テンソルに変換
    label_pos = torch.from_numpy(label_matrix).type(torch.FloatTensor).view(-1)

    # train_edgeの準備
    if train_pairs.shape[0] > 0:
        train_edge = train_pairs.copy()
        train_edge[:, 2] = 2 * train_edge[:, 2] - 1  # 0->-1, 1->1
        train_edge = np.vstack((train_edge, train_edge[:, [1, 0, 2]]))
    else:
        train_edge = np.array([]).reshape(0, 3)

    return_data = [
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        valid_mask,  # 検証マスクを追加
        test_mask,
        atom_shape,
    ]

    return tuple(return_data)
