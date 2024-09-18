# array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
SCENE=$1
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/base_partition_d --batch_type same --batch_ray_type grid --batch_size 64 --batch_until 3000 --batch_partition --batch_rays --single_reg --grid_ray_fix --batch_decrease --batch_decrease_step 500