#!/bin/bash

DIR=$1
FILE=$2
URL=$3
CHECKSUM=$4

if [ ! -d "$DIR" ]; then
  mkdir -p $DIR
fi

cd $DIR
if [ -f $FILE ]; then
  echo $FILE " already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
    rm $FILE
    exit -1
  fi
fi

echo "Downloading " $FILE " ..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."


