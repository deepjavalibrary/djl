#!/usr/bin/env bash
python training/trainCiFar10.py --epochs 10 --model resnet50_v1 --gpus 1 --mode hybrid --lr 0.001 --wd 0.001 --use_thumbnail
mv image-classification.log image-classification-symbolic.log
python training/trainCiFar10.py --epochs 10 --model resnet50_v1 --gpus 1  --mode hybrid --lr 0.001 --wd 0.001 --use-pretrained --use_thumbnail
mv image-classification.log image-classification-symbolic-pretrained.log
python training/trainCiFar10.py --epochs 10 --model resnet50_v1 --gpus 1 --mode imperative --lr 0.001 --wd 0.001 --use_thumbnail
mv image-classification.log image-classification-imperative.log
