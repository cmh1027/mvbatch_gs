SCENE=$1
DEVICE=$2
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py -s data/mipnerf360/${SCENE} -m output/${SCENE}/batch16 --config configs/mipnerf360/${SCENE}.json --vis_iteration_interval 30000 --batch_size 16