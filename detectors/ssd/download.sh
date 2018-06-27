echo "download model graph : Detectors"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget $( extract_download_url http://www.mediafire.com/file/ivwws1znd4y2v9y/graph_mobilenet_v2_fddb_180627.pb ) -O graph_mobilenet_v2_fddb_180627.pb
