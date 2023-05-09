#!/bin/bash

python src/explain.py data=organmnist3d.yaml xai_method=voxel.yaml

python src/explain.py data=vesselmnist3d.yaml xai_method=voxel.yaml