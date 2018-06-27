# Training for Face Detection

## Datasets

- FDDB
- WiderFace

## Models

- SSD + Inception v2
- SSD + Mobilenet v2

### FDDB Validset

| Model | Training<br/>Dataset | mAP | mAP<br/>(large) | mAP<br/>(medium) | mAP<br/>(small) | mAP<br/>@.50IOU | mAP<br/>@.75IOU | Recall/AR@1 | AR@10 | AR@100 | AR@100<br/>(large) | AR@100<br/>(medium) | AR@100<br/>(small) | classification<br/>loss | localization<br/>loss |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| SSD + Mobilenet v2 | FDDB | 0.714 | 0.788 | 0.690 | 0.242 | 0.962 | 0.848 | 0.446 | 0.763 | 0.768 | 0.829 | 0.750 | 0.410 | 2.978 | 0.194 |
| SSD + Mobilenet v2 | All  | 0.578 | 0.646 | 0.566 | 0.213 | 0.954 | 0.641 | 0.364 | 0.649 | 0.652 | 0.698 | 0.646 | 0.286 | 2.446 | 0.403 |
| | | | | | | | | | | | | | | | 
| SSD + Inception v2 | FDDB | 0.648 | 0.761 | 0.589 | 0.321 | 0.971 | 0.731 | 0.406 | 0.716 | 0.721 | 0.818 | 0.678 | 0.432 | 6.272 | 0.546 |

### WiderFace Validset

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

## Export a checkpoint to a graph


```
$ python tensorflow-models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor --output_directory ./models/ \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix ${TRAIN_PATH}
```


## Reference

- Tensorflow Object Detection API
  - https://github.com/tensorflow/models/blob/master/research/object_detection/
  - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
  