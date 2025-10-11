# global variables
notation="AdamW_lr0.01"
model_name="DMMV_A"
random_seed=2021
is_training=1

# Train parameters
optimizer="AdamW"
train_epochs=50
patience=10
batch_size=8
learning_rate=0.01
device_ids="0"

# dataset
root_path_name="./dataset/weather"
data_path_name="weather.csv"
data_name="custom"
period=144
features="M"
enc_in=21

# model parameters
individual=1
norm_const=0.4
align_const=0.4
vm_arch="mae_base"
ft_type="none"


# Start training
for pred_len in 96 192 336 720; do
    history_len=336

    exp_info="${model_name}_${data_path_name%.*}_${pred_len}_${notation}"
    trained_MAE_ckpt="./checkpoints/mae_visualize_vit_base.pth" # Need to change to your path of pretrained MAE model in same datasets

    python -u run.py \
        --exp_info "$exp_info" \
        --model "$model_name" \
        --random_seed "$random_seed" \
        --is_training "$is_training" \
        --optimizer "$optimizer" \
        --train_epochs "$train_epochs" \
        --patience "$patience" \
        --batch_size "$batch_size" \
        --learning_rate "$learning_rate" \
        --devices_ids "$device_ids" \
        --root_path "$root_path_name" \
        --data_path "$data_path_name" \
        --data "$data_name" \
        --period "$period" \
        --features "$features" \
        --c_in "$enc_in" \
        --history_len "$history_len" \
        --pred_len "$pred_len" \
        --individual "$individual" \
        --norm_const "$norm_const" \
        --align_const "$align_const" \
        --vm_arch "$vm_arch" \
        --ft_type "$ft_type" \
        --trained_MAE_ckpt "$trained_MAE_ckpt" 
done