![image](https://github.com/dontlearncpp/Body-size-measurement-with-CowK-Net/assets/103402250/d40da654-6735-45ee-8a45-436664c63cd0)

The keypoints was detected with CowK-Net, and the body size was measured with 6points.py  
# CowK-Net
## Train
--output_dir "logs/coco_r50" -c config/edpose.cfg.py --options batch_size=4 epochs=60 lr_drop=55 num_body_points=6 backbone=resnet50 --dataset_file="coco"
## eval 
--output_dir "logs/coco_r50" -c logs/coco_r50/config_cfg.py --options batch_size=4 epochs=60 lr_drop=55 num_body_points=6 backbone=resnet50 --dataset_file="coco" --pretrain_model_path="logs/coco_r50/checkpoint_best_regular.pth" --eval
## test and the output of keypoints
Changing the util/visualizer.py with visualizer-test.py, the name shoule be unchange.
Using this file, the coordinate of predicted keypoints will be written in test.txt
## Environment Setup
Install Pytorch and torchvision
pip install -r requirements.txt
# Measurement of cow body size
IIn the 6points.py the body size was automated measured.
1. Specify the folder to store depth images and RGB images -Line 261
2. Enter image name -Line 231
3. Enter keypoint coordinates -Line 230


