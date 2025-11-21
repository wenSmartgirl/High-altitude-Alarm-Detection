#!/bin/bash

build_dir=host_ubuntu_build

# 删除编译目录
rm -rf ${build_dir}

# 重新创建编译目录
mkdir ${build_dir}

# 进入编译目录
pushd ${build_dir}

# 配置, 此处可以利用 -D 添加编译选项
cmake -DWITH_NCNN=OFF -DWITH_RKNN=ON .. 

# 编译
make -j

# 安装
make install 

# 退出目录
popd

