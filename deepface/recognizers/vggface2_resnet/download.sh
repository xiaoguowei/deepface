#!/bin/bash

echo "[download] model graph : VGGFace-Resnet"
DIR="$(cd "$(dirname "$0")" && pwd)"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget $( extract_download_url http://www.mediafire.com/file/datqeencsrqua22/rcmalli_vggface_tf_resnet50.h5 ) -O $DIR/weight.h5
wget $( extract_download_url http://www.mediafire.com/file/sc2fo5xcktwbqkd/rcmalli_vggface_labels_v2.npy ) -O $DIR/labels.npy

wget $( extract_download_url http://www.mediafire.com/file/a550w53sqykwos2/saved_model_180814.tar.gz ) -O $DIR/saved_model.tar.gz
tar -xvf saved_model.tar.gz