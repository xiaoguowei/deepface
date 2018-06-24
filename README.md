# Training for Face Detection

## Datasets

- FDDB
- WiderFace

## Models

- SSD + Inception v2

## Dependencies & Install

- tensorflow >= 1.8.0
- opencv >= 3.0
- protobuf compiler >= 3.0

Other python packages can be installed using requirements.txt, 

```
$ pip install -r requirements.txt
```

Also protobufs for object detection should be compiled.

```
$ cd tensorflow-models/research/
$ protoc object_detection/protos/*.proto --python_out=.
$ cd -
```

## Train

Training a new model requires two steps.

- Create/Modify a config file. eg. ./config/ssd_inception_v2_fddb.config
- Train a model using commands as below.

```
$ export PYTHONPATH=$PYTHONPATH:$PWD/tensorflow-models/research:$PWD/tensorflow-models/research/slim
$ python tensorflow-models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=./configs/ssd_inception_v2_fddb.config \
    --train_dir=./checkpoints/ssd_inception_v2_fddb
```

Evaluation for the trained model can be executed as below.

```
$ python tensorflow-models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=./configs/ssd_inception_v2_fddb.config \
    --checkpoint_dir=./checkpoints/ssd_inception_v2_fddb/ \
    --eval_dir=./evaluation/
```
