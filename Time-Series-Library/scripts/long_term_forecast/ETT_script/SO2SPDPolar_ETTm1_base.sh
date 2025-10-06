#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=SO2SPDPolar

# Fast & effective configuration based on SOTA papers
# Key changes: d_model 128, lr 0.001, dropout 0.1, larger batch
# 7 channels, 4 prediction lengths: 96, 192, 336, 720

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 3 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 3 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 3 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 3 \
  --des 'Exp' \
  --itr 1
