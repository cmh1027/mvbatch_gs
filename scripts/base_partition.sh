# array=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
SCENE=$1
INTERVAL=100
EXIT=10000
python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/base_part1 
# python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/base_part2 --single_partial_rays --single_partial_rays_denom 2 --log_single_partial --log_single_partial_interval ${INTERVAL} --test_iteration_interval ${INTERVAL} --forced_exit ${EXIT} --turn_off_print
# python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/base_part4 --single_partial_rays --single_partial_rays_denom 4 --log_single_partial --log_single_partial_interval ${INTERVAL} --test_iteration_interval ${INTERVAL} --forced_exit ${EXIT} --turn_off_print
# python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/base_part8 --single_partial_rays --single_partial_rays_denom 8 --log_single_partial --log_single_partial_interval ${INTERVAL} --test_iteration_interval ${INTERVAL} --forced_exit ${EXIT} --turn_off_print
# python train.py --source_path data/mipnerf360/${SCENE} --config configs/${SCENE}.json -m output/${SCENE}/base_part16 --single_partial_rays --single_partial_rays_denom 16 --log_single_partial --log_single_partial_interval ${INTERVAL} --test_iteration_interval ${INTERVAL} --forced_exit ${EXIT} --turn_off_print