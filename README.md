# UniRec-Vision
## 一体化视觉识别系统（人脸 + 物品 + 语音播报）
**英文名称**：UniRec-Vision
**中文名称**：统一视觉识别平台

基于 YOLOv8 + InsightFace + MobileNetV2 + Edge-TTS 构建的实时视频流多目标识别系统，同时支持人脸识别、通用物品识别、本地特征注册、持久化特征库、智能语音播报。项目采用多线程解耦架构，画面流畅不卡顿，全平台兼容 Windows / Linux / WSL2。

---

## 一、项目简介
本项目将**人脸、物品**两类目标统一管理到一套本地特征库，核心能力如下：
1. 支持本地USB摄像头、IP摄像头/手机DroidCam视频流实时检测
2. 人脸、物品分离特征提取，配置独立相似度阈值，兼顾识别精度与防误检
3. 一键空格键完成目标特征注册，特征向量以 `.npz` 文件本地持久化存储
4. 基于 Edge-TTS 实现中文语音播报，内置时间间隔限制，避免重复播报
5. 多线程任务解耦：图像采集、推理识别、语音播放相互独立，不阻塞主线程
6. 自动适配多系统中文字体，画面中文正常渲染
7. 显存不足时自动降级至CPU运行，低配设备也可正常使用

**适用场景**：智能视觉演示、简易智能播报、本地离线人脸识别/物品识别、学习二次开发。

---

## 二、技术栈
| 功能模块 | 技术选型 | 作用说明 |
| ---- | ---- | ---- |
| 目标检测 | YOLOv8-n | 视频画面中定位人脸、各类物品目标 |
| 人脸识别 | InsightFace (buffalo_l) | 人脸特征提取、人脸相似度比对 |
| 物品特征提取 | MobileNetV2 | 通用物体特征向量化，完成相似度匹配 |
| 语音合成 | Edge-TTS | 在线文字转中文语音 |
| 图像渲染 | OpenCV + PIL | 视频解码、画面绘制、中文文字渲染 |
| 深度学习框架 | PyTorch / TorchVision | 模型推理、张量运算 |
| 并发处理 | Python Thread + Queue | 多线程解耦，队列控流防止卡顿 |
| 数据存储 | Numpy `.npz` | 特征向量本地文件存储 |

---

## 三、项目目录结构
UniRec-Vision/
├── main.py # 主程序源码（核心运行文件）
├── requirements.txt # Python 依赖清单
├── .gitignore # Git 忽略配置（过滤缓存、虚拟环境、临时文件）
├── README.md # 项目完整文档（本文档）
├── unified_database/ # 程序自动生成：人脸 / 物品特征库（*.npz 文件）
└── /tmp/sent_tts.mp3 # 运行临时音频文件，程序退出自动删除


> 补充说明：
> 1. `unified_database` 目录首次运行程序自动创建，存放所有注册的特征数据；
> 2. 临时音频文件仅运行期间产生，正常退出会自动清理。

---

## 四、环境要求
### 1. 基础运行环境
- Python 版本：`3.8 ~ 3.11`（推荐 3.10，兼容性最佳）
- 操作系统：Windows 10/11、Ubuntu、Linux 发行版、WSL2
- 硬件：纯CPU可运行（速度较慢）；推荐 NVIDIA 显卡 + CUDA 加速

### 2. 系统级额外依赖
#### Linux / WSL2 系统（必须安装）
1. 音频播放器（无此组件无法播放语音）
```bash
sudo apt update
sudo apt install mpg123

中文字体（解决画面中文乱码 / 方框问题）
bash
运行
sudo apt install fonts-wqy-zenhei

# 常规安装
pip install -r requirements.txt

# 国内镜像加速（网络慢时使用）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
