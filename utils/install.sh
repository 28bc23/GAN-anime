#!/bin/bash

echo -e "For what purpose do you want to install this program?:\n t - training\n g - olny generating"
read -p "[T/p]: " V

V=${V:-t}
V=$(echo "$V" | tr '[:upper:]' '[:lower:]')

if [ "$V" = "p" ]; then
  echo "Installing GAN for generating only ..."
else
  echo "Installing GAN ..."
  
  DIR_T="$HOME/.gan"

  echo "Set directory for installation"
  read -p "[$DIR_T]: " DIR

  DIR=${DIR:-$DIR_T}
  mkdir -p $DIR

  if [ ! -d "$DIR" ]; then
    echo "Directory don't exist"
    exit 1
  fi

  
  echo "Directory of installation: $DIR"

  git clone https://github.com/28bc23/GAN-anime.git $DIR
  
  DATA_DIR="$DIR/data"
  mkdir -p $DATA_DIR
  if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory doesn't exist"
    exit 1
  fi
  wget -O "$DATA_DIR/dataset.zip" https://huggingface.co/datasets/skytnt/fbanimehq/resolve/main/data/fbanimehq-00.zip?download=true

  if ! command -v unzip >/dev/null 2>&1
  then
    echo "UNZIP COULD NOT BE FOUND, PLEASE INSTALL IT"
    exit 1
  fi

  unzip "$DATA_DIR/dataset.zip" -d "$DATA_DIR"
  rm "$DATA_DIR/dataset.zip"

fi
