# ALL scripts in this file come from Autoformer
export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for model_name in peak_Autoformer
do 
for pred_len in 120 240 360 720
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_Autoformer_720_$pred_len \
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
        --busy_ratio 0.5 \
        --learning_rate 1e-5 \
        --des 'Exp' \
        --itr 1  >logs/LongForecasting/$model_name'_Etth1_'$pred_len'ours.log'
  
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small \
        --data_path ETTh2.csv \
        --model_id ETTh2_Autoformer_720_$pred_len \
        --model $model_name \
        --data ETTh2 \
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
        --learning_rate 1e-5 \
        --busy_ratio 0.5 \
        --des 'Exp' \
        --itr 1  >logs/LongForecasting/$model_name'_Etth2_'$pred_len'ours.log'

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id electricity_Autoformer_720_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 720 \
        --label_len 360 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --busy_ratio 0.5 \
        --learning_rate 1e-4 \
        --des 'Exp' \
        --itr 1 >logs/LongForecasting/$model_name'_electricity_'$pred_len'ours.log'

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic_Autoformer_720_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 720 \
        --label_len 360 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --busy_ratio 0.5 \
        --learning_rate 1e-4 \
        --des 'Exp' \
        --itr 1 >logs/LongForecasting/$model_name'_traffic_'$pred_len'ours.log'
done
done