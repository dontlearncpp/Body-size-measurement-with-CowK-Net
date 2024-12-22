<div align=center>
<img src="https://github.com/dontlearncpp/Body-size-measurement-with-CowK-Net/assets/103402250/d40da654-6735-45ee-8a45-436664c63cd0"> 
</div>
The keypoints was detected with CowK-Net, and the body size was measured with 6points.py 

# CowK-Net
<div align=center>
<img src="https://github.com/dontlearncpp/Body-size-measurement-with-CowK-Net/assets/103402250/83e25383-d69d-4ff8-a989-bfee48b05fea"> 
</div>

## Train
--output_dir "logs/coco_r50" -c config/edpose.cfg.py --options batch_size=4 epochs=60 lr_drop=55 num_body_points=6 backbone=resnet50 --dataset_file="coco"
## eval 
--output_dir "logs/coco_r50" -c logs/coco_r50/config_cfg.py --options batch_size=4 epochs=60 lr_drop=55 num_body_points=6 backbone=resnet50 --dataset_file="coco" --pretrain_model_path="logs/coco_r50/checkpoint_best_regular.pth" --eval
## test and the output of keypoints
Changing the util/visualizer.py with visualizer-test.py, the name shoule be unchange.
Using this file, the coordinate of predicted keypoints will be written in test.txt
## Tese on CowDatabase and CowDatabase1
This model also trained and test on those two datasets.
The image can be download at:
* [Cowdatabase](https://github.com/ruchaya/CowDatabase)
* [Cowdatabase2](https://github.com/ruchaya/CowDatabase2)
* the coco format json files for Cowdatabase are avaliable at [Cowdatabase-coco](https://drive.google.com/file/d/1CugDe6dXkmw5hxtO7DVvb_-DrxBQfB0C/view?usp=drive_link)
* the coco format json files for Cowdatabase2 are avaliable at [Cowdatabase2-coco](https://drive.google.com/file/d/1gTnl22uGgKwvFZSY_Gaclfuytua3Ar8e/view?usp=drive_link)
  
The rgb images captured from left Kinect camera are applied for train and test.
The json file with 6 points are avelibale.

## Environment Setup
Install Pytorch and torchvision

pip install -r requirements.txt
# Measurement of cow body size
IIn the 6points.py the body size was automated measured.
1. Specify the folder to store depth images and RGB images -Line 261
2. Enter image name -Line 231
3. Enter keypoint coordinates -Line 230


