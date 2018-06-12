echo "download model graph : keras vggface"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget $( extract_download_url http://www.mediafire.com/file/datqeencsrqua22/rcmalli_vggface_tf_resnet50.h5 ) -O weight.h5
wget $( extract_download_url http://www.mediafire.com/file/sc2fo5xcktwbqkd/rcmalli_vggface_labels_v2.npy ) -O labels.npy