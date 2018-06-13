echo "download model graph : VGGFace1"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget $( extract_download_url http://www.mediafire.com/file/j8aqfjojwl29c5m/weight.mat/file ) -O weight.mat