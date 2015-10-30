#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
FETCH_SCRIPT=$BASE_DIR/scripts/fetch_data.sh
DATA_DIR=$BASE_DIR/data

FILE=mav_videos.tar.gz
URL="https://www.dropbox.com/s/o0e00y3dfr35by2/mav_videos.tar.gz?dl=0"
CHECKSUM=33c65bb24a8a631f5c005e9be61ccb65
$FETCH_SCRIPT $DATA_DIR $FILE $URL $CHECKSUM
