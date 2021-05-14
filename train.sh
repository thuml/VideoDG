#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_val.py \
    something RGB \
    --mode train \
    --train_list data_list/ucf_train.txt \
    --val_list data_list/ucf_val.txt \
    --test_list "data_list/hmdb_train.txt data_list/hmdb_val.txt" \
    --num_class 12 \
    --arch resnet50 \
    --num_segments 5 \
    --consensus_type APN \
    --batch-size 40 \
    --snapshot_pref test \
    --lr 0.001 \
    --gd 20 \
    --alpha 0.01
