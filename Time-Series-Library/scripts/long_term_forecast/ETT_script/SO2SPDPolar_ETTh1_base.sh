#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=SO2SPDPolar

# Improved configuration for ETTh1 dataset
# Changes: dropout 0.1->0.3, d_model 512->384, warmup reduced to 2 epochs
# 7 channels, 4 prediction lengths: 96, 192, 336, 720

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --n_heads 8 \
  --dropout 0.3 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --n_heads 8 \
  --dropout 0.3 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --n_heads 8 \
  --dropout 0.3 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --n_heads 8 \
  --dropout 0.3 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --des 'Exp' \
  --itr 1
