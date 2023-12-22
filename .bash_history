wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000263/data/data.tar.gz
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000263/data/code.tar.gz
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000263/data/code.tar.gz
tar -zxvf data.tar.gz
tar -zxvf code.tar.gz
pwd
~/code/v2# conda create --name myenv --clone base
~/code/v2 conda create --name myenv --clone base
cd code/v2
nvidia -smi
nvidia-smi
df -h
conda create --name myenv --clone base
conda activate myenv
conda deactivate
pwd
cd ../.
cd ..
conda activate myenv
pwd
cd code/v2
conda activate myenv
conda init
nvidia-smi
nvidia-smi -l 0.5
nvidia-smi -l 1
conda activate myenv
pip install matplotlib tensorboard pandas jupyter
pwd
df -h
import torch
python
nvidia-smi
git init
apt install git
git iniit
git init
gs
touch .gitignore
git add .gitignore
git commit -m ".gitignore commit"
git config --global user.email "nhw2417@gmail.com"
git config --global user.name "Hyunwoo Nam"
git commit -m ".gitignore commit"
git push
git remote add origin https://github.com/nhw2417/boostcamp_level1_project.git
git push -u origin master
git push -u origin master
git push -u origin master
git status
git add .
gti status
git status
git commit -m "first commit"
git status
git push
git config --global credential.helper store
git push
git push
git status
git add .
git status
git commit -m 'gitignore .tar.gz file'
git push
git status
git add .
git commit -m 'second commit'
git push
git rm --cached data.tar.gz
pwd
git rm --cached code.tar.gz
git rm --cached
git rm data.tar.gz
git add
git status
git push
git log
git reset HEAD^
git status
git lgo
git log
git reset HEAD^
git log
git reset HEAD^
git status
git add .
git commit -m 'first commit'
git push
export
export SM_CHANNEL_TRAIN=/data/ephemeral/home/train/images/
export SM_CHANNEL_EVAL=/data/ephemeral/home/eval
echo SM_CHANNEL_TRAIN
echo
echo $SM_CHANNEL_TRAIN
pwd
cd code/v2
cd ~
ls
pwd
pip install tensorboard
tensorboard --logdir moel
tensorboard --logdir model
python code/v2/train.py --name "focal loss"
python code/v2/inference.py 
python code/v2/inference.py --model_dir "./model/focal loss"
tensorboard --logdir model
git add .
git commit -m 'focal loss'
git push
cd code/v2
python train.py --name "basecode v2"
python inference.py --model_dir "basecode v2"
python inference.py --model_dir "model/basecode v2"
tesorboard --logdir model
tensorboard --logdir model
git add .
git commit -m 'basecode v2'
git push
rm -r ../../model
git add .
git commit -m 'remove directory with location mistake'
git add .
git status
git ad
git add
tensorboard --logdir model
python train.py --name "focal loss" --criterion focal
tensorboard --logdir model
python inference.py --model_dir model/focal loss
python inference.py --model_dir model/focal_loss
python inference.py --model_dir model/focal_loss
python inference.py --model_dir model/basecode_v2
python inference.py --model_dir model/focal_loss --output_dir output/focal_loss
python inference.py --model_dir model/basecode_v2 --output_dir output/basecode_v2
git add .
git commit -m 'inference focal_loss'
git push
rm -r ../../output
watch -n 1 nvidia-smi
python
python
conda activate myenv
pwd
python code/v2/inference.py --model_dir .cache/torch/hub/checkpoints/resnet18-f37072fd.pth --output_dir code/v2/output/resnet18_pretrained
python code/v2/inference.py --model_dir .cache/torch/hub/checkpoints --output_dir code/v2/output/resnet18_pretrained
rm code/v2/output/resnet18_pretrained/
rmdir code/v2/output/resnet18_pretrained/
python code/v2/inference.py --model_dir .cache/torch/hub/checkpoints --output_dir code/v2/output/resnet18_pretrained
source /opt/conda/bin/activate myenv
source /opt/conda/bin/activate myenv
watch -n 1 nvidia-smi
source /opt/conda/bin/activate myenv
pwd
cd code/v2
python
python inference.py --model_dir ../../.cache/torch/hub/checkpoints --output_dir output/resnet18_pretrained
python inference.py --model_dir ../../.cache/torch/hub/checkpoints --output_dir output/resnet18_pretrained
python inference.py --model_dir ../../.cache/torch/hub/checkpoints --output_dir output/resnet18_pretrained
python inference.py --model_dir ../../.cache/torch/hub/checkpoints --output_dir output/resnet18_pretrained
python train.py --name pretrained_resnet50
echo $SM_CHANNEL_TRAIN
cd ~
export SM_CHANNEL_TRAIN train/images
export SM_CHANNEL_TRAIN=/data/ephemeral/home/train/images
echo $SM_CHANNEL_TRAIN
export SM_CHANNEL_EVAL=/data/ephemeral/home/eval
python train.py --name pretrained_resnet50
pwd
cd code/v2
python train.py --name pretrained_resnet50
python inference.py --model_dir "./model/pretrained_resnet50" --output_dir "./output/pretrained_resnet50"
python inference.py --model_dir "./model/pretrained_resnet50" --output_dir "./output/pretrained_resnet50"
cd ..
cd custum
cd custom/
git clone https://github.com/nhw2417/boostcamp_level1_project/tree/master/code/v2
git add .
git add ~
git add ../../.
git add .
echo $SM_MODEL_DIR
echo $SM_CHANNEL_TRAINI
echo $SM_CHANNEL_TRAIN
--name help
help --name
python train.py --name resnet50 --model ResNet50
python train.py --name resnet50 --model ResNet50
python inference.py --model_dir model/resnet502 --output output/resnet50
python inference.py --model_dir model/resnet50 --output output/resnet50
python inference.py --model_dir model/resnet50 --output output/resnet50 --model ResNet50
python train.py --name resnet50_batch256 --model ResNet50 --batch_size 256
python inference.py --model_dir model/resnet50_batch256 --output output/resnet50_batch256 --model ResNet50
python
source /opt/conda/bin/activate myenv
watch -l 1 nvidia-smi
watch -n 1 nvidia-smi
source /opt/conda/bin/activate myenv
watch -n 1 nvidia-smi
conda activate myenv
cd code/custom/
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
python train.py --model ResNet50 --criterion focal --name resnet50_focal
tensorboard --logdir model
python train.py --model ResNet50 --criterion label_smoothing --name resnet50_labelsmoothing
echo $SM_CHANNEL_EVAL
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
python inference.py --model Resnet50 --model_dir "model/resnet_f1" --output_dir "output/resnet_f1"
python inference.py --model ResNet50 --model_dir "model/resnet_f1" --output_dir "output/resnet_f1"
python inference.py --model ResNet50 --model_dir "model/resnet50_f1" --output_dir "output/resnet_f1"
python inference.py --model ResNet50 --model_dir "model/resnet50_focal" --output_dir "output/resnet_focal"
python inference.py --model ResNet50 --model_dir "model/resnet50_labelsmoothing" --output_dir "output/resnet50_labelsmoothing"
py
python train.py --model ResNet50 --name resnet50
tensorboard --logdir
tensorboard --logdir model
pip install torchmetrics
python train.py --model ResNet50 --name resnet50
python train.py --model ResNet50 --name resnet50
python train.py --model ResNet50 --name resnet50
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
tensorboard --logdir model
python inference.py --model ResNet50 --model_dir model/resent50 --ourput_dir output/resnet50
python inference.py --model ResNet50 --model_dir model/resent50 --output_dir output/resnet50
python inference.py --model ResNet50 --model_dir model/resnet50 --output_dir output/resnet50
tensorboard --logdir model
python train.py --batch_size 512 --model ResNet50 --name batchsize_test
python train.py --batch_size 1024 --model ResNet50 --name batchsize_test2
python train.py --batch_size 32 --model ResNet50 --name batchsize_test3
python train.py --model ResNet50 --name batchsize_test4
tensorboard --logdir model
tensorboard --logdir model
python train.py --model ResNet50 --name batchsize_test5
tensorboard --logdir model
python inference.py --model ResNet50 --model_dir model/resnet50_batch32 --output_dir output/resnet50_batch32
python train.py --model ResNet50 --criterion f1 --name resnet50_f1
tensorboard --logdir model
python train.py --model ResNet50 --criterion focal --name resnet50_focal
tensorboard --logdir model
python train.py --model ResNet50 --optimizer AdamW --name resnet50_admaw
tensorboard --logdir model
python
python train.py --model WideResNet101 --optimizer AdamW --name wideresnet101_adamw
python train.py --model WideResNet101 --optimizer AdamW --name wideresnet101_adamw
python train.py --model WideResNet101 --optimizer AdamW --name wideresnet101_adamw
python train.py --model WideResNet101 --optimizer AdamW --name wideresnet101_adamw
tensorboard --logdir
tensorboard --logdir model
python train.py --model ResNext101 --optimizer AdamW --name resnext101_adamw
tensorboard --logdir model
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw
python
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw --resize [224, 224]
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw --resize 224, 225
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw --resize 224, 224
python train.py --model VitB16 --optimizer AdamW --name vitb16_adamw --resize 224 224
tensorboard --logdir model
python train.py --model WideResNet101 --optimizer AdamW --name wideresnet101_adamw
git add .
git add .
tensorboard --logdir model
git pull
git push
python train.py --model ResNext101 --optimizer AdamW --criterion focal --name resnext101_adamw_focal
tensorboard --logdir model
python train.py --model ResNext101 --optimizer AdamW --criterion label_smoothing --name resnext101_adamw_labelsmoothing
tensorboard --logdir mdoel
tensorboard --logdir mdoel
tensorboard --logdir model
python train.py --model ResNext101 --optimizer AdamW --criterion f1 --name resnext101_adamw_f1
python train.py --model ResNext101 --optimizer AdamW --name resnext101_adamw_extrafc
tensorboard --logdir model
git commit -am 'add model'
git status
git add .
git add .
git add .
git status
git commit -m 'add model'
git push
