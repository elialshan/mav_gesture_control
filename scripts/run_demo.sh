#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
BIN_DIR=$BASE_DIR/build
MAV_VIDEO_DIR=$BASE_DIR/data/mav_videos

$BIN_DIR/tools/demo.bin -i $MAV_VIDEO_DIR/aeroquad01.mp4 -m 1
