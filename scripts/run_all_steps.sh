#!/bin/bash

python src/main_explain.py data=vesselmnist3d.yaml explain_method=volume.yaml

python src/main_eval.py data=vesselmnist3d.yaml explain_method=volume.yaml eval_method=volume_vessel.yaml attr_path='explain_VesselMNIST3D.npz'

python src/main_rank.py