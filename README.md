# deepface

![blackpink with deepface(vgg model)](./etc/example_blackpink.png)

Deep Learning Models for Face Detection/Recognition/Alignments, implemented in Tensorflow.

It is being implemented...

## Main Contributers

@ildoonet @hahahahannie

## Models

### Baseline

A baseline model use dlib face detection module to crop rois. Then they will be forwarded at the vggface network to extract features.

- dlib face detection
- dlib face alignment
- VGGFace face recognition

## Experiments

### LFW Dataset

| Model                                 | Set        | 1-EER      | Accuracy |
|---------------------------------------|------------|------------|----------|
| VGG(Paper, No Embedding, Trained)     | Test       | 0.9673     |          |
| VGG(Paper, Embedding, Trained)        | Test       | 0.9913     |          |
|                                       |            |            |          |
| VGG(no embedding, no training on lfw) | Test       | 0.9400     | 0.936    |
| VGG2-Resnet(no training on lfw)       | Test       | 0.9680     | 0.949    |

## Install

### Requirements

- tensorflow >= 1.8.0
- opencv >= 3.4.1

### Install & Download Models

```bash
$ pip install -r requirements.txt
$ cd detectors/dlib
$ bash download.sh
$ cd ../../recognizers/vggface/
$ bash download.sh
$ cd ../resnet/
$ bash download.sh
```

## Run

### Test on samples

```bash
$ python bin/run_example.py run --source_path=./samples/faces --db_path=./sample_db.pkl --img_path=./samples/blackpink/blackpink1.jpg --method=vgg2
```

### Test on a image

```bash
$ python bin/face.py run --visualize=true --image=./samples/blackpink/blackpink1.jpg
```

## Reference

### Models

[1] VGG Face : http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

[2] VGG Face in Tensorflow : https://github.com/ZZUTK/Tensorflow-VGG-face

[3] DLib : https://github.com/davisking/dlib

[4] Dlib Guide Blog : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

[5] VGG Face 2 Project : https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

[6] Kera VGG Face2 : https://github.com/rcmalli/keras-vggface

### Datasets

[1] LFW : http://vis-www.cs.umass.edu/lfw/
