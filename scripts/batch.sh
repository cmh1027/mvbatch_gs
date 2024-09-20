array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
BATCH=$1
for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch${BATCH} --batch_type sample --batch_sample_strategy max --batch_ray_type random --batch_size ${BATCH} --batch_until 9000 --batch_partition --batch_rays --single_reg --partial_ssim
done