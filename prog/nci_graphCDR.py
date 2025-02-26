import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from model import *
from my_utiils import *
from nci_data_load import dataload
from nci_data_process import process
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(description="Drug_response_pre")
parser.add_argument("--alph", dest="alph", type=float, default=0.30, help="")
parser.add_argument("--beta", dest="beta", type=float, default=0.30, help="")
parser.add_argument("--epoch", dest="epoch", type=int, default=350, help="")
parser.add_argument(
    "--hidden_channels", dest="hidden_channels", type=int, default=256, help=""
)
parser.add_argument(
    "--output_channels", dest="output_channels", type=int, default=100, help=""
)
args = parser.parse_args()
start_time = time.time()

print("Loading data files...")
# ------data files
(
    drug_feature,
    mutation_feature,
    gexpr_feature,
    methylation_feature,
    nb_celllines,
    nb_drugs,
) = dataload()

train = pd.read_csv("../nci_data/train.csv")
train["labels"] = np.load("../nci_data/train_labels.npy")

val = pd.read_csv("../nci_data/val.csv")
val["labels"] = np.load("../nci_data/val_labels.npy")

test_data = pd.read_csv("../nci_data/test.csv")
test_data["labels"] = np.load("../nci_data/test_labels.npy")

train["labels"] = train["labels"].astype(int)
val["labels"] = val["labels"].astype(int)
test_data["labels"] = test_data["labels"].astype(int)

print("Processing train/test split...")
# -------split train and test sets
(
    drug_set,
    cellline_set,
    train_edge,
    label_pos,
    train_mask,
    valid_mask,
    test_mask,
    atom_shape,
) = process(
    drug_feature,
    mutation_feature,
    gexpr_feature,
    methylation_feature,
    train,
    val,
    test_data,
    nb_celllines,
    nb_drugs,
)

print("Test mask positive examples:", torch.sum(label_pos[test_mask] > 0).item())
print("Test mask total examples:", torch.sum(test_mask).item())

print("Initializing model...")
model = GraphCDR(
    hidden_channels=args.hidden_channels,
    encoder=Encoder(args.output_channels, args.hidden_channels),
    summary=Summary(args.output_channels, args.hidden_channels),
    feat=NodeRepresentation(
        atom_shape,
        gexpr_feature.shape[-1],
        methylation_feature.shape[-1],
        args.output_channels,
    ),
    index=nb_celllines,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
myloss = nn.BCELoss()


def train():
    model.train()
    loss_temp = 0
    print("Training batch:", end=" ")
    for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
        print(f"{batch+1}", end=" ")
        optimizer.zero_grad()
        pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
            drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
        )
        dgi_pos = model.loss(pos_z, neg_z, summary_pos)
        dgi_neg = model.loss(neg_z, pos_z, summary_neg)
        pos_loss = myloss(pos_adj[train_mask], label_pos[train_mask])
        loss = (
            (1 - args.alph - args.beta) * pos_loss
            + args.alph * dgi_pos
            + args.beta * dgi_neg
        )
        loss.backward()
        optimizer.step()
        loss_temp += loss.item()
    print("\nTrain loss: ", str(round(loss_temp, 4)))


# 検証関数の追加
def validate():
    model.eval()
    print("Validating...")
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            _, _, _, _, pre_adj = model(
                drug.x,
                drug.edge_index,
                drug.batch,
                cell[0],
                cell[1],
                cell[2],
                train_edge,
            )

            loss_temp = myloss(pre_adj[valid_mask], label_pos[valid_mask])

            # 予測値と真の値を取得
            yp = pre_adj[valid_mask].detach().numpy()
            yvalid = label_pos[valid_mask].detach().numpy()

            # 評価指標の計算
            AUC, AUPR, F1, ACC = metrics_graph(yvalid, yp)

            print("Validation loss: ", str(round(loss_temp.item(), 4)))
            print("Validation metrics:")
            print(" AUC: " + str(round(AUC, 4)))
            print(" AUPR: " + str(round(AUPR, 4)))

            return AUC, AUPR, F1, ACC


def test():
    model.eval()
    print("Testing...")
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            _, _, _, _, pre_adj = model(
                drug.x,
                drug.edge_index,
                drug.batch,
                cell[0],
                cell[1],
                cell[2],
                train_edge,
            )
            loss_temp = myloss(
                pre_adj[test_mask], torch.tensor(test_data["labels"]).float()
            )

        # 予測値と真の値を取得
        yp = pre_adj[test_mask].detach().numpy()
        ytest = test_data["labels"]

        # 連続値の評価指標（AUC, AUPR）
        AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)

        # 二値分類のための閾値処理
        yp_binary = (yp > 0.5).astype(int)

        # 追加の評価指標を計算
        accuracy = accuracy_score(ytest, yp_binary)
        precision = precision_score(ytest, yp_binary)
        recall = recall_score(ytest, yp_binary)
        f1 = f1_score(ytest, yp_binary)

        print("Test loss: ", str(round(loss_temp.item(), 4)))
        print("Test metrics:")
        print("  AUC: " + str(round(AUC, 4)))
        print("  AUPR: " + str(round(AUPR, 4)))
        print("  F1: " + str(round(F1, 4)))
        print("  ACC: " + str(round(ACC, 4)))
        print("  Precision: " + str(round(precision, 4)))
        print("  Recall: " + str(round(recall, 4)))

    return AUC, AUPR, F1, ACC, precision, recall, accuracy


import os

# Variables to store model state
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 結果を保存するディレクトリを作成
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ------main
print("\nStarting training...")
final_metrics = {"AUC": 0, "AUPR": 0, "F1": 0, "ACC": 0, "Precision": 0, "Recall": 0}

# 出力ディレクトリの設定
output_dir = getattr(args, "o", "./results/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CSVファイルのパス
test_results_path = os.path.join(output_dir, "test_results.csv")
file_exists = os.path.isfile(test_results_path)

# ファイルが存在しない場合はヘッダーを書き込む
if not file_exists:
    with open(test_results_path, "w") as f:
        f.write("ACC,Precision,Recall,F1,AUC,AUPR\n")

# メインループの修正
best_valid_auc = 0
best_model_state = None

for epoch in range(args.epoch):
    print("\nEpoch: " + str(epoch + 1) + "/" + str(args.epoch))
    train()
    valid_AUC, valid_AUPR, valid_F1, valid_ACC = validate()

    # 検証データでの性能が向上した場合にモデルを保存
    if valid_AUC > best_valid_auc:
        best_valid_auc = valid_AUC
        best_model_state = model.state_dict().copy()
        print("New best model found!")

# 最良のモデルを読み込んでテスト
print("\nLoading best model for final evaluation...")
model.load_state_dict(best_model_state)
AUC, AUPR, F1, ACC, precision, recall, accuracy = test()

# 結果の出力
print("\n" + "=" * 40)
print("Evaluation completed!")
print("\nFinal test metrics:")
print(" AUC: " + str(round(AUC, 4)))
print(" AUPR: " + str(round(AUPR, 4)))
print(" F1: " + str(round(F1, 4)))
print(" ACC: " + str(round(ACC, 4)))
print(" Precision: " + str(round(precision, 4)))
print(" Recall: " + str(round(recall, 4)))
print("=" * 40)

# 結果をCSVに保存
with open(test_results_path, "a") as f:
    f.write(f"{ACC},{precision},{recall},{F1},{AUC},{AUPR}\n")
