# array=("stump" "bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "treehill")
# BATCH=$1
# for (( i=0; i<${#array[@]}; i++ ));
# do
#     SCENE=${array[$i]}
#     python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch${BATCH} --init_type sfm --batch_type sample --batch_sample_strategy max --batch_ray_type random --batch_size ${BATCH} --batch_until 9000 --batch_partition --batch_rays --single_reg --partial_ssim
# done


array=("room" "bicycle" "flowers" "treehill")
for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch4_end --init_type sfm --batch_size 4
done


