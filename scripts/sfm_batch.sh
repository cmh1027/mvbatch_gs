
if [ $1 = 0 ]; then
    array=("bicycle" "bonsai" "counter")
elif [ $1 = 1 ]; then
    array=("room" "stump" "treehill")
elif [ $1 = 2 ]; then
    array=("flowers" "garden" "kitchen")
fi

for (( i=0; i<${#array[@]}; i++ ));
do
    SCENE=${array[$i]}
    python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch4_grid256 --batch_size 4 --batch_partition --mask_grid --mask_height 16 --mask_width 16
done

