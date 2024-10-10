SCENE="bicycle"
BATCH="4"
python train.py -s data/mipnerf360/${SCENE} -m output/${SCENE}/debug --config configs/mipnerf360/${SCENE}.json --vis_iteration_interval 30000 --num_imgs 10 --start_checkpoint output/bicycle/chkpnt3000.pth