#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
FETCH_SCRIPT=$BASE_DIR/scripts/fetch_data.sh

FILE=models.tar.gz
URL="https://www.dropbox.com/s/tfqi5ypvbxwihyr/models.tar.gz?dl=0"
CHECKSUM=afb76af08bff3f026ba18adcb5dca5f8
$FETCH_SCRIPT $BASE_DIR $FILE $URL $CHECKSUM

