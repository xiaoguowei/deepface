#!/bin/bash

cd ../../models/research
python object_detection/train.py --logtostderr --pipeline_config_path=/home/kabrain2/Downloads/deepface/custom_models/ssd_inception_v2/ssd_inception_v2_fddb.config --train_dir=/home/kabrain2/Downloads/deepface/custom_models/ssd_inception_v2/train
