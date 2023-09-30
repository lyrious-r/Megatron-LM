# syntax=docker/dockerfile:1.2
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
  && apt-get update \
  && apt-get install -y python3-pip python3-dev git curl sudo wget ccache libz-dev \
                        libllvm-8-ocaml-dev libllvm8 llvm-8 llvm-8-dev             \
                        llvm-8-doc llvm-8-examples llvm-8-runtime                  \
                        clang-8 clang-tools-8 clang-8-doc libclang-common-8-dev    \
                        libclang-8-dev libclang1-8 clang-format-10                 \
                        python-clang-8 libfuzzer-8-dev lldb-8 lld-8                \
                        libc++-8-dev libc++abi-8-dev libomp-8-dev clang-tidy-8     \
                        openssh-server vim libtool autoconf libgflags-dev          \
                        libgoogle-glog-dev libcairo2-dev redis

# update glibc
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt install --only-upgrade libstdc++6 -y

RUN curl -s -L https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.sh -o cmake.sh \
  && sh cmake.sh --skip-license --prefix=/usr/local/

# install aws efa software
ARG EFA_INSTALLER_VERSION=latest
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify

ENV MOFED_VER 5.3-1.0.0.1
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH

## Install NCCL
RUN git clone https://github.com/NVIDIA/nccl /opt/nccl \
    && cd /opt/nccl \
    && git checkout v2.12.12-1 \
    && make -j src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"

RUN cd /usr/local/bin && ln -s /usr/bin/python3 python

ENV HOME /root

## Install python dependencies
RUN python3 -m pip install --upgrade pip
# torch nightly
RUN python3 -m pip install --pre torch==2.1.0.dev20230621+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117
RUN python3 -m pip install torchvision --no-deps
# apex
# apex has weird installation issue. see https://github.com/NVIDIA/apex/issues/1594
RUN python3 -m pip install setuptools packaging
RUN cd $HOME \
    && git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && git checkout 0da3ffb \
    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# deepspeed
# disable aio and sparse attn (incompatible with triton in torch nightly)
RUN cd $HOME \
    && git clone -b mlm_exp https://github.com/chenyu-jiang/DeepSpeed.git \
    && cd DeepSpeed \
    && DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_SPATIAL_INFERENCE=0 pip install -e .

# sed through source code to remove torch._six import
RUN cd $HOME \
    && python3 -c "from pathlib import Path; import deepspeed; f = open(\"./ds_path.txt\", \"w\"); f.write(str(Path(deepspeed.__file__).parent)); f.close();" \
    && DEEPSPEED_PATH=`cat ds_path.txt` \
    && find $DEEPSPEED_PATH -name '*.py' -exec sed -i 's/from torch\._six import inf/from torch import inf/g' {} +

# required for elkai (used to solve TSP in dynapipe)
RUN python3 -m pip install scikit-build

# install DynaPipe
RUN cd $HOME \
    && git clone https://github.com/chenyu-jiang/DynaPipe.git \
    && cd DynaPipe/ \
    && pip3 install -r requirements.txt \
    && pip3 install -e .

# make extensions
RUN cd $HOME/DynaPipe \
    && cd dynapipe/data_opt \
    && make \
    && cd ../memory_opt \
    && python3 setup.py build

# Clone Megetron-LM and install apex
RUN cd $HOME \
    && git clone -b dynapipe https://github.com/chenyu-jiang/Megatron-LM

RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR nltk
RUN TMPDIR=/var/tmp python3 -m pip install --upgrade requests
RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR pybind11 jsonlines pillow pandas matplotlib seaborn

# install nsight systems
RUN apt-get install -y cuda-nsight-systems-11-7

RUN echo "$(echo -n 'source /env.sh\n'; cat ~/.bashrc)" > ~/.bashrc

COPY ./env.sh .
COPY ./entrypoint /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint
ENTRYPOINT ["entrypoint"]