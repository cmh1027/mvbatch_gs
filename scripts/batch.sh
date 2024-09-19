
SCENE=$1
BATCH=$2
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch${BATCH}_mean --batch_type sample --batch_sample_strategy max --batch_ray_type random --batch_size ${BATCH} --batch_until 9000 --batch_partition --batch_rays --single_reg --batch_grad_mean


