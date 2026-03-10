if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=PatchMixer

root_path_name=./dataset/jepx
data_path_name=jepx2022-2026.csv
model_id_name=JEPX
data_name=custom

random_seed=2021

# test plot config
plot_every_n_batches=20
plot_num_features=4
plot_sample_idx=0
# Optional explicit feature names, e.g. "system_price" or "system_price,hokkaido_price"
plot_features="system_price,area_price_chubu"

# enc_in = number of feature columns (excluding 'date')
# Current base schema: system_price + 9 area prices + sell/buy bid + contract + 4 block volumes = 17
# If congestion columns are present, increase accordingly.
enc_in=17

for pred_len in 48 96 192 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_M_sl'$seq_len'_pl'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --target system_price \
      --seq_len $seq_len \
      --pred_len $pred_len \
    --plot_every_n_batches $plot_every_n_batches \
    --plot_num_features $plot_num_features \
    --plot_sample_idx $plot_sample_idx \
    --plot_features "$plot_features" \
      --enc_in $enc_in \
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --freq t \
      --des 'Exp' \
      --train_epochs 200 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --loss_flag 2 \
      --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_fM_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_seed'$random_seed.log
done
