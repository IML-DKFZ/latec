#!/bin/bash

python src/explain.py data=modelnet40.yaml xai_method=point_cloud.yaml

python src/explain.py data=shapenet.yaml xai_method=point_cloud.yaml

python src/explain.py data=coma.yaml xai_method=point_cloud.yaml