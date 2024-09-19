array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
UNTIL=$1
for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch_decrease_u${UNTIL} --batch_type sample --batch_sample_strategy max --batch_ray_type random --batch_size 64 --batch_until 9000 --batch_partition --batch_rays --single_reg --batch_grad_mean --batch_decrease --batch_decrease_step 1000 --batch_decrease_until ${UNTIL}
done

# 0 64
# 500 32
# 1000 16
# 1500 8
# 2000 4
# 2500 2
# 3000 1

# 0 64
# 1000 32
# 2000 16
# 3000 8
# 4000 4
# 5000 2
# 6000 1ß