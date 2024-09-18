# array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")

# for (( i=0; i<${#array[@]}; i++ ));
# do
#     python train.py --source_path data/mipnerf360/${array[$i]} --config configs/${array[$i]}.json -m output/${array[$i]}/batch${BATCH} --batch_type sample --batch_sample_strategy random --batch_size ${BATCH} --batch_until 9000 --batch_partition --batch_rays
# done
SCENE=$1
BATCH=$2
UNTIL=$3
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch${BATCH}_mean_it${UNTIL} --batch_type sample --batch_sample_strategy max --batch_ray_type random --batch_size ${BATCH} --batch_until ${UNTIL} --batch_partition --batch_rays --single_reg --batch_grad_mean
