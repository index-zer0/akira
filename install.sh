#!/bin/bash

if ! [ -x "$(command -v gcc)" ]; then
  echo 'gcc is not installed. Please use "sudo apt install build-essential" and start the restart the installation' >&2
  exit 1
fi

if ! [ -x "$(command -v git)" ]; then
  echo 'git is not installed. Please use "sudo apt install git" and start the restart the installation.' >&2
  exit 1
fi
DIR=$(pwd)

if ! [ -d "$DIR/cmatrix" ]; then
  git clone https://github.com/index-zer0/cmatrix.git
fi

if ! [ -d "$DIR/MNIST_for_C" ]; then
  git clone https://github.com/takafumihoriuchi/MNIST_for_C/
fi

cd $DIR/cmatrix && git reset --hard bfa3136
cd $DIR/MNIST_for_C && git reset --hard 77405bb

PATH_TO_DATA="$DIR/MNIST_for_C/data"
cd $DIR/MNIST_for_C && sed -i "s|\.\/data|$PATH_TO_DATA|g" mnist.h

cd $DIR && make clean && make && cat $DIR/public/a-ascii.txt && echo "--- akira installed successfully ---" && exit 0