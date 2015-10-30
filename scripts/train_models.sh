#! /bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

if [ -d $BASE_DIR/new_models ]; then
  rm -r $BASE_DIR/new_models
fi

python $BASE_DIR/scripts/train.py --train_data_dir $BASE_DIR/data --train_output_dir $BASE_DIR/tmp --models_dir $BASE_DIR/new_models --mav_videos_dir $BASE_DIR/data/mav_videos
