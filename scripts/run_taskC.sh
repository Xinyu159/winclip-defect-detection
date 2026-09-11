#!/bin/bash
# 任务 C:GPU 真实节拍(P50/P95)。**必须在 GPU 空闲时跑** ——
# 与别的任务并发会把排队时间算进时延,测出来的数字是假的。
#
# 测三档,全部 CUDA(同硬件;CPU 数字在本地另有,分表):
#   fp32-CUDA / fp16-CUDA / int8-CUDA
# TensorRT 未安装(见 T01_结果.md),本脚本不测。
#
# 每档:
#   A. 端到端整图(patcher + 整图塔 + 两尺度全量窗口)= research 完整路径
#   B. 窗口预算档位 4/8/16/32/64/128 = 级联的实际形态
# 预热 10 帧,计时 50 次,取 P50/P95(产线看 P95,不看均值)。
cd /root/autodl-tmp/winclip || exit 1
PY=/root/miniconda3/envs/winclip/bin/python
ROOT=/root/autodl-tmp/mvtec_anomaly_detection

$PY -u scripts/bench_onnx.py \
    --data_root $ROOT \
    --text data/deploy_onnx_dyn/text_protos \
    --classes tile,bottle \
    --configs "fp32cuda=data/deploy_onnx_dyn:CUDAExecutionProvider,fp16cuda=data/deploy_onnx_fp16:CUDAExecutionProvider,int8cuda=data/deploy_onnx_int8:CUDAExecutionProvider" \
    --n-warm 10 --n-iter 50 --n-e2e-frames 10 --n-budget-frames 25 \
    --budgets 4,8,16,32,64,128 \
    --out /root/autodl-tmp/winclip/logs/bench_taskC.json \
    > /root/autodl-tmp/winclip/taskC.log 2>&1
echo "[exit] bench = $?"

# CPU 档单独一张表(任务书硬规则:不许混硬件)。ORT 的动态量化目标就是 CPU,
# 所以 int8-CPU 是 int8 的**正常**部署形态,CUDA 那张只是为同硬件对比而测。
$PY -u scripts/bench_onnx.py \
    --data_root $ROOT \
    --text data/deploy_onnx_dyn/text_protos \
    --classes tile,bottle \
    --configs "fp32cpu=data/deploy_onnx_dyn:CPUExecutionProvider,int8cpu=data/deploy_onnx_int8:CPUExecutionProvider" \
    --n-warm 10 --n-iter 50 --n-e2e-frames 10 --n-budget-frames 25 \
    --budgets 4,8,16,32,64,128 \
    --out /root/autodl-tmp/winclip/logs/bench_taskC_cpu.json \
    > /root/autodl-tmp/winclip/taskC_cpu.log 2>&1
echo "[exit] bench_cpu = $?"

echo "ALL_DONE"
