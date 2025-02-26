#!/bin/bash

# 実行回数を設定
NUM_RUNS=5

# 各実行の結果を表示するための関数
run_experiment() {
    run_number=$1
    echo "Starting run ${run_number} of ${NUM_RUNS}"
    python nci_graphCDR.py
    echo "Completed run ${run_number}"
    echo "----------------------------------------"
}

# メインの実行ループ
for i in $(seq 1 $NUM_RUNS)
do
    run_experiment $i
done

echo "All runs completed!"
