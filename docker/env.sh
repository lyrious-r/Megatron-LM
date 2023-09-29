#!/bin/bash
export CUDA_HOME=/usr/local/cuda-11.7
export NCCL_ROOT_DIR=/opt/nccl/build
export NCCL_DIR=/opt/nccl/build
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=/opt/nvidia/nsight-systems/2022.1.3/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH