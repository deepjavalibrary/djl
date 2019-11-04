#!/usr/bin/env bash
python training/trainCiFar10.py --epochs 10 --model resnet50_v2 --mode symbolic --lr 0.001 --wd 0.001 --use-pretrained False
mv image-classification.log image-classification-symbolic.log
python training/trainCiFar10.py --epochs 10 --model resnet50_v2 --mode symbolic --lr 0.001 --wd 0.001 --use-pretrained True
mv image-classification.log image-classification-symbolic-pretrained.log
python training/trainCiFar10.py --epochs 10 --model resnet50_v2 --mode imperative --lr 0.001 --wd 0.001 --use-pretrained False
mv image-classification.log image-classification-imperative.log
