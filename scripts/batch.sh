array=("stump" "bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "treehill")
BATCH=$1
for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch${BATCH}_random --batch_sample_strategy random --batch_size ${BATCH} --batch_partition
done

