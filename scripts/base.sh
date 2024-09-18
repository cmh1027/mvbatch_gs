array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
# array=("bicycle" "bonsai" "counter")
# array=("flowers" "garden" "kitchen")
# array=("room" "stump" "treehill")
for (( i=0; i<${#array[@]}; i++ ));
do
    python train.py --source_path data/mipnerf360/${array[$i]} --config configs/${array[$i]}.json -m output/${array[$i]}/base
done


