#!/bin/bash

python src/main_explain.py data=modelnet40.yaml explain_method=point_cloud.yaml

python src/main_eval.py data=coma.yaml explain_method=point_cloud.yaml eval_method=point_cloud_coma.yaml attr_path=' explain_coma.npz

python src/main_rank.py