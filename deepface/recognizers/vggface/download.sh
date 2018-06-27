#!/bin/bash

echo "[download] model graph : VGGFace1"
DIR="$(cd "$(dirname "$0")" && pwd)"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget -c --tries=2 $( extract_download_url http://www.mediafire.com/file/j8aqfjojwl29c5m/weight.mat/file ) -O $DIR/weight.mat
echo "[download] end"
#rm $DIR/shape_predictor_68_face_landmarks.dat.bz2