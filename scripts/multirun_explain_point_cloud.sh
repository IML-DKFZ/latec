#!/bin/bash

python src/main/main_explain.py data=modelnet40.yaml explain_method=point_cloud.yaml

python src/main/main_explain.py data=shapenet.yaml explain_method=point_cloud.yaml

python src/main/main_explain.py data=coma.yaml explain_method=point_cloud.yaml