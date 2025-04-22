# R-ABLE

files above 25 mb in https://drive.google.com/drive/folders/16fjjc-vDgBavxZp5YxXrSk0nYWzIgVVX?usp=sharing

```bash
#training command
python -u main.py --dataset cifar10_imb  --data-dir './data' --workers 0 --num-class 10 --pmodel_path './pmodel/cifar10.pt' --arch resnet18 --temperature 0.1 --loss_weight 1.0 --cuda_VISIBLE_DEVICES '0' --epochs 500 --batch-size 64 --lr 0.01 --wd 1e-3 --cosine --seed 123
