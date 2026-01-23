# Tạo file run_exp.sh với nội dung sau:
#!/bin/bash

# Nhận tham số seed từ bên ngoài (nếu không nhập thì mặc định là 1706)
SEED=${1:-1706}
CONFIG="configs/rtdetr/rtdetr_r18vd_6x_cityscapes.yml"

echo "--------------------------------"
echo "STARTING EXPERIMENT WITH SEED: $SEED"
echo "--------------------------------"

# 1. Lệnh Train
python tools/train.py -c $CONFIG --amp --seed $SEED 2>&1 | tee cityscapes_train_log_${SEED}.txt

# 2. Lệnh Test
# (Lưu ý: Check xem file checkpoint có sinh ra chưa)
CKPT="output/rtdetr_r18vd_6x_cityscapes/checkpoint0071.pth"

if [ -f "$CKPT" ]; then
    echo "Found checkpoint. Starting evaluation..."
    python tools/train.py -c $CONFIG -r $CKPT --test-only 2>&1 | tee cityscapes_test_log_${SEED}.txt
else
    echo "Checkpoint not found: $CKPT"
fi