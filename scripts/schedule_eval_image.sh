#!/bin/bash

python src/eval.py data=oct.yaml attr_path=/attr_oct_dataset_14_methods_2023-04-20_14-32-42.npz

python src/eval.py

python src/eval.py data=resisc45.yaml attr_path=/attr_resisc45_dataset_14_methods_2023-04-20_14-35-32.npz