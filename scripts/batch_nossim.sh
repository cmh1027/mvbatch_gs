BATCH=$1
array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
for (( i=0; i<${#array[@]}; i++ ));
do
    python train.py --source_path data/mipnerf360/${array[$i]} --config configs/${array[$i]}.json -m output/${array[$i]}/batch${BATCH}_nossim --batch_type sample --batch_sample_strategy random --batch_size ${BATCH} --batch_until 9000 --batch_partition --batch_rays  --lambda_dssim 0
done

