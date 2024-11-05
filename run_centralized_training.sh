#!/bin/bash

# run python script with arguments
python3 centralized_baseline.py --dataset "CIFAR10" --n_epochs 100 --batch_size 32 --learning_rate 0.001 --early_stopping_rounds 10