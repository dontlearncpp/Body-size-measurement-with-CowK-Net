The keypoints was detected with CowK-Net, and the body size was measured with 6points.py  
# CowK-Net
## Train
--output_dir "logs/coco_r50" -c config/edpose.cfg.py --options batch_size=4 epochs=60 lr_drop=55 num_body_points=6 backbone=resnet50 --dataset_file="coco"
## eval 
--output_dir "logs/coco_r50" -c logs/coco_r50/config_cfg.py --options batch_size=4 epochs=60 lr_drop=55 num_body_points=6 backbone=resnet50 --dataset_file="coco" --pretrain_model_path="logs/coco_r50/checkpoint_best_regular.pth" --eval
## test and the output of keypoints
change the util/visualizer.py with visualizer.py 
Using this file, the coordinate of predicted keypoints will be written in test.txt
# 

