#!/bin/bash

# Install Fields2Cover
sudo apt-get update
sudo apt-get install -y --no-install-recommends software-properties-common
sudo add-apt-repository -y ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential ca-certificates cmake \
     doxygen g++ git libeigen3-dev libgdal-dev libpython3-dev python3 python3-pip \
     lcov libgtest-dev libtbb-dev swig libgeos-dev gnuplot libtinyxml2-dev nlohmann-json3-dev
python3 -m pip install gcovr

git clone https://github.com/Rick-v-E/Fields2Cover/ ~/Fields2Cover
# git clone https://github.com/Fields2Cover/Fields2Cover ~/Fields2Cover
cd ~/Fields2Cover && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TUTORIALS=OFF -DBUILD_TESTS=OFF -DUSE_ORTOOLS_RELEASE=ON -DBUILD_PYTHON=ON ..
make -j$(nproc)
sudo make install

# Install GNOME-terminal 
sudo apt-get install -y --no-install-recommends gnome-terminal

# Install Latex
sudo apt-get install -y --no-install-recommends texlive dvipng texlive-latex-extra texlive-fonts-recommended cm-super
 
# Install Python dependencies
sudo apt-get install -y --no-install-recommends python3-tk
python3 -m pip install --upgrade pip
python3 -m pip install -e /workspaces/drone-grid-env
