array=("stump" "bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "treehill")
for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}s
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm --init_type sfm
done


