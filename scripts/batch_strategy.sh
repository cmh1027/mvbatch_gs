SCENE=$1
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch4_min --init_type sfm --batch_size 4 --batch_sample_strategy min
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/sfm_batch4_random --init_type sfm --batch_size 4 --batch_sample_strategy random

