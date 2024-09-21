SCENE=$1
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch${BATCH}_random --batch_sample_strategy random --batch_size ${BATCH} --batch_until 30000
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/batch${BATCH}_min --batch_sample_strategy min --batch_size ${BATCH} --batch_until 30000
