SCENE=$1
BATCH=$2
DEVICE=$3
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py -s data/mipnerf360/${SCENE} -m output/${SCENE}/batch${BATCH} --config configs/mipnerf360/${SCENE}.json --vis_iteration_interval 30000 --batch_size ${BATCH}