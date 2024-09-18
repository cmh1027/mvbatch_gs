array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
for (( i=0; i<${#array[@]}; i++ ));
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm --init_type sfm
done


