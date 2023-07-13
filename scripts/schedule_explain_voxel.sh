#!/bin/bash

python src/explain.py data=organmnist3d.yaml explain_method=voxel.yaml

python src/explain.py data=vesselmnist3d.yaml explain_method=voxel.yaml

python src/explain.py data=adrenalmnist3d.yaml explain_method=voxel.yaml