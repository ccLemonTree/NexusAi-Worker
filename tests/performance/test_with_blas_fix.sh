#!/bin/bash
# 测试设置 BLAS 单线程后的性能

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "已设置 BLAS 单线程环境变量:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo ""

echo "测试 1: NumPy 操作"
python3 test_numpy_ops.py

echo ""
echo "测试 2: 完整 yolov5() 函数"
python3 test_yolov5_direct.py --image test.png --concurrent 16
