"""部署运行时包:纯 numpy + OpenVINO(+PIL/cv2),零 torch/open_clip 依赖。

ov_engine   — IR 加载/编译 + open_clip 同款 preprocess(逐位镜像)
pipeline    — 镜像 winclip.WinCLIP 打分结构(窗口索引/调和/few-shot,可对拍)
roi_cv      — 传统 CV 前置:CLAHE + patch 统计建模 + 马氏距离 ROI 窗口建议
registry    — 冻结空间缺陷原型注册器(stage/commit/query)
"""
