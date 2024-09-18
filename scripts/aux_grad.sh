SCENE=$1
THRES=$2
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch4_aux_thres${THRES} --batch_type sample --batch_sample_strategy max --batch_ray_type random --batch_size 4 --batch_until 9000 --batch_partition --batch_rays --single_reg --batch_grad_mean --aux_densify --aux_densify_threshold ${THRES}
