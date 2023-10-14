# ALL scripts in this file come from Autoformer
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/abla" ]; then
    mkdir ./logs/LongForecasting/abla
fi

for model_name in peak_DLinear
do 
for pred_len in 120 240 360 
do
for alpha in 0
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_Autoformer_720_$pred_len'_'$alpha \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 720 \
        --label_len 360 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --busy_ratio $alpha \
        --learning_rate 1e-4 \
        --des 'Exp' \
        --itr 1  >logs/LongForecasting/abla/$model_name'_ETTh1_'$pred_len'_'$alpha.log
done
done
done