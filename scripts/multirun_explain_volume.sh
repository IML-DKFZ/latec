#!/bin/bash

python src/main/main_explain.py data=organmnist3d.yaml explain_method=volume.yaml

python src/main/main_explain.py data=vesselmnist3d.yaml explain_method=volume.yaml

python src/main/main_explain.py data=adrenalmnist3d.yaml explain_method=volume.yaml