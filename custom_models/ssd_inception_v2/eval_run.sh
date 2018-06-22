#!/bin/bash

cd ../../../
python object_detection/eval.py --logtostderr --pipeline_config_path=/home/kabrain2/tensorflow_model/research/object_detection/models/ssd_inception_v2/ssd_inception_v2_fddb.config --checkpoint_dir=/home/kabrain2/tensorflow_model/research/object_detection/models/ssd_inception_v2/train --eval_dir=/home/kabrain2/tensorflow_model/research/object_detection/models/ssd_inception_v2/eval
