#!/bin/bash
# 任务 B:int8 量化端到端漂移。fp32 与 int8 **同在 CUDA** 上测 —— 只有
# 同硬件对比,差值才能归因于量化本身(任务书硬规则:不许混硬件)。
#
# 两个刻意的写法(都是踩过的坑):
#   1. **不接 grep**。上一版把输出管进 `grep -v`,grep 按块缓冲,进程没退出
#      就一个字都不吐 —— 表现为"跑了 30 分钟毫无进展",实际在正常计算;
#      更糟的是中途 kill 会把整个缓冲区丢掉。现在直接写文件。
#   2. eval_ov.py 已改为**逐类落盘**,任一类算完立刻可见,断开/崩溃不丢整轮。
#
# int8 在 CUDA 上比 fp32 慢约 8×(QInt8 MatMul 无 CUDA 内核,回落 CPU 执行
# 并伴随 H2D/D2H 拷贝)。这是**真实部署结论**,不是脚本问题:ORT 的动态量化
# 目标是 CPU EP,int8 该上 CPU 跑。此处为满足"同硬件对比"仍放 CUDA 测。
cd /root/autodl-tmp/winclip || exit 1
PY=/root/miniconda3/envs/winclip/bin/python

echo "########## fp32 (CUDA) ##########"
$PY -u scripts/eval_ov.py \
    --data_root /root/autodl-tmp/mvtec_anomaly_detection \
    --deploy data/deploy_onnx_dyn --text data/deploy_onnx_dyn/text_protos \
    --classes all --shots 0,4 --device cuda --tag fp32_cuda \
    > /root/autodl-tmp/winclip/taskB_fp32.log 2>&1
echo "[exit] fp32 = $?"

echo "########## int8 (CUDA) ##########"
$PY -u scripts/eval_ov.py \
    --data_root /root/autodl-tmp/mvtec_anomaly_detection \
    --deploy data/deploy_onnx_int8 --text data/deploy_onnx_int8/text_protos \
    --classes all --shots 0,4 --device cuda --tag int8_cuda \
    > /root/autodl-tmp/winclip/taskB_int8.log 2>&1
echo "[exit] int8 = $?"

echo "ALL_DONE"
