#!/bin/bash

conda activate [ENV]
set -x
set -e
echo "Starting..."

for conv in gat gcn
do
  for dataset in citeseer wikics computers photo physics ppi
  do
    hidden_channels=4

    python -u train_teacher.py --dataset $dataset

    training=supervised
    expt_name=${dataset}_supervised
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --conv $conv
    expt_name=${dataset}_supervised_de
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --drop_edge --conv $conv


    training=kd
    expt_name=${dataset}_kd_dd
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --do_drop --conv $conv
    expt_name=${dataset}_kd_dd_de
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --do_drop --drop_edge --conv $conv

    training=kd
    expt_name=${dataset}_kd
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --conv $conv
    expt_name=${dataset}_kd_de
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --drop_edge --conv $conv

    training=nce
    expt_name=${dataset}_nce_de
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --drop_edge --conv $conv
    expt_name=${dataset}_nce
    python -u gnn.py --training $training --expt_name $expt_name --dataset $dataset --hidden_channels $hidden_channels --conv $conv

  done
done

$SHELL