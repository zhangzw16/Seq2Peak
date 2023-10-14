# ALL scripts in this file come from Autoformer
export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/alpha" ]; then
    mkdir ./logs/LongForecasting/alpha
fi

for model_name in Autoformer
do 
for pred_len in 120 240 360 720
do
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
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
        --learning_rate 1e-5 \
        --des 'Exp' \
        --itr 1  >logs/LongForecasting/alpha/$model_name'_Etth1_'$pred_len'_'$alpha.log
done
done
done