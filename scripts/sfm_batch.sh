array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
BATCH=$1
TYPE=$2
for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch${BATCH}_${TYPE} --batch_sample_strategy ${TYPE} --batch_size ${BATCH} --batch_partition --init_type sfm
done

