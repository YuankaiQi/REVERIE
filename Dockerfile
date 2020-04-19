  
# Matterport3DSimulator
# Requires nvidia gpu with driver 384.xx or higher


FROM nvidia/cudagl:9.0-devel-ubuntu16.04

# Install a few libraries to support both EGL and OSMESA options
# ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get remove python3.5
RUN apt-get remove --auto-remove python3.5
RUN apt-get purge python3.5
RUN apt-get purge --auto-remove python3.5
#python-software-properties
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.6
# switch python version
RUN rm /usr/bin/python
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.6 /usr/bin/python3
RUN ln -s /usr/bin/python3.6 /usr/bin/python
# 
RUN apt-get install -y  wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev 
RUN apt-get install -y  libglew-dev libopencv-dev python-opencv python3-setuptools python3.6-dev python3-pip python3.6-tk
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install backports.functools-lru-cache==1.4 cycler==0.10.0 decorator==4.1.2 matplotlib==2.1.0 networkx==2.0
RUN pip3 install numpy==1.18.2 olefile pandas==0.21.0 Pillow>=4.3.0 pyparsing==2.2.0 python-dateutil==2.6.1
RUN pip3 install pytz==2017.3 pyyaml>=4.2b1 six==1.11.0 scipy==1.2.1 nltk scikit-image
RUN pip3 install opencv-python Cython easydict tensorboardX cffi h5py


# find suitable whl file at https://download.pytorch.org/whl/cu90/torch_stable.html
RUN wget https://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install ./torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision==0.1.8
RUN rm ./torch-0.4.0-cp36-cp36m-linux_x86_64.whl

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build
