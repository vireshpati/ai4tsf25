#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=SO2SPDPolar

# Grid search hyperparameters
seq_lens=(96)
pred_lens=(96)
d_models=(128 256)
n_heads_list=(4 8)
e_layers_list=(1 2 3)
learning_rates=(0.0001 0.001)
dropouts=(0.1)
batch_sizes=(32 64)

# ETTh1 dataset has 7 channels
enc_in=7
dec_in=7
c_out=7

for seq_len in "${seq_lens[@]}"; do
  label_len=$((seq_len / 2))

  for pred_len in "${pred_lens[@]}"; do
    for d_model in "${d_models[@]}"; do
      for n_heads in "${n_heads_list[@]}"; do
        for e_layers in "${e_layers_list[@]}"; do
          for lr in "${learning_rates[@]}"; do
            for dropout in "${dropouts[@]}"; do
              for batch_size in "${batch_sizes[@]}"; do

                echo "Running: seq_len=$seq_len, pred_len=$pred_len, d_model=$d_model, n_heads=$n_heads, e_layers=$e_layers, lr=$lr, dropout=$dropout, batch_size=$batch_size"

                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/ETT-small/ \
                  --data_path ETTh1.csv \
                  --model_id ETTh1_${seq_len}_${pred_len} \
                  --model $model_name \
                  --data ETTh1 \
                  --features M \
                  --seq_len $seq_len \
                  --label_len $label_len \
                  --pred_len $pred_len \
                  --e_layers $e_layers \
                  --d_layers 1 \
                  --enc_in $enc_in \
                  --dec_in $dec_in \
                  --c_out $c_out \
                  --d_model $d_model \
                  --n_heads $n_heads \
                  --dropout $dropout \
                  --batch_size $batch_size \
                  --learning_rate $lr \
                  --train_epochs 100 \
                  --patience 10 \
                  --des 'GridSearch' \
                  --itr 1

              done
            done
          done
        done
      done
    done
  done
done
