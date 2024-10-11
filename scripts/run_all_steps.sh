#!/bin/bash

python src/main/main_explain.py data=vesselmnist3d.yaml explain_method=volume.yaml

python src/main/main_eval.py data=vesselmnist3d.yaml explain_method=volume.yaml eval_metric=volume_vessel.yaml attr_path='saliency_maps_vesselmnist3d.npz'

python src/main/main_rank.py