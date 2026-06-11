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

Windows 系统
无需额外安装系统组件，字体、命令自动兼容。
##五、依赖文件：requirements.txt
在项目根目录新建 requirements.txt，粘贴以下内容：
txt
# 基础图像处理库
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0

# YOLOv8 目标检测
ultralytics>=8.2.0

# 人脸识别 & ONNX 运行库
insightface>=0.7.3
onnxruntime>=1.16.0

# 深度学习框架
torch>=2.0.0
torchvision>=0.15.0

# 语音合成
edge-tts>=6.1.9
安装依赖命令
bash
运行
# 常规安装
pip install -r requirements.txt

# 国内镜像加速（网络慢时使用）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
## 六、Git 忽略配置：.gitignore
在项目根目录新建 .gitignore，粘贴以下内容，避免上传缓存、虚拟环境、临时文件：
gitignore
# Python 编译缓存
__pycache__/
*.pyc
*.pyo
*.pyd
*.pyx

# 虚拟环境目录
venv/
env/
.venv/
ENV/

# 模型权重文件
*.pt
*.pth
*.onnx
*.trt

# 音视频临时文件
*.mp3
*.wav
*.mp4
/tmp/

# 本地特征库（默认不上传，如需同步可删除此行）
unified_database/

# 系统垃圾文件
.DS_Store
Thumbs.db
desktop.ini

# 日志文件
*.log

# IDE 配置文件
.vscode/
.idea/
*.swp
*.swo
## 七、项目配置与运行
1. 核心参数配置
打开 main.py，顶部全局参数可根据自身设备、场景修改：
python
运行
# 同一目标最小识别间隔(秒)，防止重复推理
RECOG_INTERVAL = 1.5
# 同一名称语音播报最小间隔(秒)，防止语音刷屏
TTS_INTERVAL = 3.0
# YOLO 推理分辨率，数值越小速度越快、精度略降
YOLO_SIZE = 320
# 人脸匹配相似度阈值（越低越宽松，越高越严格）
FACE_SIM_THRESHOLD = 0.45
# 物品匹配相似度阈值（物品易误检，默认设置偏高）
OBJ_SIM_THRESHOLD = 0.80
# 队列最大长度，控制内存占用
MAX_QUEUE_SIZE = 3
# 自动选择 CUDA / CPU 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 视频流地址：0=本地USB摄像头；IP地址=网络摄像头/DroidCam
CAM_URL = "http://192.168.1.6:4747/video"
常用修改场景
本地 USB 摄像头：CAM_URL = 0
画面卡顿：降低 YOLO_SIZE、增大 RECOG_INTERVAL
物品误识别：调高 OBJ_SIM_THRESHOLD
语音播报频繁：调高 TTS_INTERVAL
2. 启动程序
bash
运行
python main.py
3. 按键操作说明
表格
按键	功能
空格键 (SPACE)	截取画面中第一个检测到的目标，执行特征注册，控制台输入名称即可完成保存
ESC 键	正常退出程序，自动释放摄像头、线程、队列、临时文件
4. 画面标识说明
红色矩形框：未注册目标，标签显示 Unknown + 相似度
绿色矩形框：已注册目标，标签显示 名称 + 相似度，自动触发语音播报

##八、整体架构设计
1. 多线程分工（解耦设计，保障流畅）
主线程：读取视频流、画面绘制、按键监听、特征注册逻辑
推理子线程：YOLO 目标检测、人脸 / 物品特征提取、特征库比对（耗时操作隔离）
TTS 语音线程：语音合成、音频播放，完全不阻塞视觉流程
2. 完整数据流
plaintext
摄像头视频流 
  → 帧队列(frame_queue) 
    → 推理线程(检测 + 特征提取 + 比对) 
      → 结果队列(result_queue) 
        → 主线程绘制画面 + 推送语音任务 
          → TTS线程完成语音播报
3. 特征库存储规则
存储目录：unified_database/
文件格式：自定义名称.npz
存储内容：特征向量 + 类型标记（0 = 人脸，1 = 物品）
加载逻辑：程序启动自动遍历目录，加载全部特征数据
##九、常见问题 & 全套解决方案
1. 依赖相关问题
问题：ModuleNotFoundError 找不到模块
现象：运行提示 cv2、ultralytics、torch 等模块不存在
解决：重新执行依赖安装命令，多 Python 环境请确认使用当前环境的 pip：
bash
运行
pip install -r requirements.txt --force-reinstall
问题：MobileNetV2 预训练权重下载失败
现象：网络异常导致模型权重无法在线下载
解决：切换正常网络重新运行；内网离线环境可手动下载权重放置到本地缓存。
2. CUDA & 显存问题
问题：显存溢出 Out of memory
现象：启动人脸模型时报显存不足
解决：代码已内置自动降级逻辑，显存不足会自动切换 CPU 运行，无需手动修改；
手动优化：降低 det_size=(224,224) 为 (160,160)，关闭其他占用显存的软件。
问题：检测不到 CUDA，始终使用 CPU 运行
现象：控制台打印 DEVICE=cpu，运行速度慢
解决：安装与本机 CUDA 版本匹配的 PyTorch；执行以下代码验证 CUDA 可用性：
python
运行
import torch
print(torch.cuda.is_available())
输出 True 代表显卡加速正常。
3. 摄像头 / 视频流问题
问题：无法打开视频流
本地 USB 摄像头：将 CAM_URL = 0；检查摄像头是否被微信、浏览器等软件占用；Linux 赋予摄像头权限：
bash
运行
sudo chmod 666 /dev/video0
IP 摄像头 / DroidCam：确保手机与电脑处于同一局域网，核对 IP、端口，关闭手机防火墙与后台限制。
问题：画面卡顿、延迟高
解决：减小 YOLO_SIZE、增大 RECOG_INTERVAL；WSL2 环境检查 USB 摄像头穿透配置。
4. 中文显示乱码（方框 / 问号）
现象：画面中文无法正常显示
解决：
Linux/WSL2 安装文泉驿字体：sudo apt install fonts-wqy-zenhei
Windows 系统缺失黑体字体时，补充系统字体即可；代码已自动适配多路径字体。
5. 语音播报问题
问题：Linux 无语音、不发声
解决：确认已安装 mpg123：sudo apt install mpg123
问题：语音连续重复播报
解决：调大参数 TTS_INTERVAL = 4.0，延长播报间隔。
问题：Edge-TTS 合成音频失败
解决：保证设备正常联网；Linux 给 /tmp 目录开放权限：chmod 777 /tmp/
问题：Windows 提示 mpg123 不是内部命令
说明：代码已做异常捕获，该报错不影响主程序运行，仅语音功能失效，可忽略。
6. 识别 & 注册精度问题
问题：注册人脸后识别不到
解决：降低 FACE_SIM_THRESHOLD（如改为 0.40）；注册时保证光线充足、人脸正对镜头。
问题：物品频繁误识别
解决：调高物品阈值 OBJ_SIM_THRESHOLD = 0.85。
问题：按空格提示「未检测到目标」
解决：将目标放置在画面中心，保证 YOLO 可以完整框选目标后再执行注册。
问题：同名注册会覆盖旧特征
说明：当前逻辑同名直接覆盖 .npz 文件，如需防覆盖可自行增加名称判断逻辑。
7. 进程 / 资源释放异常
问题：关闭程序后摄像头、进程残留
说明：代码自带优雅退出逻辑，正常按 ESC 退出会清空队列、等待线程结束、删除临时文件；
手动处理：异常强杀进程后，在任务管理器 / 终端结束 Python 残留进程。
问题：WSL2 退出后摄像头无法重连
解决：重启 WSL2 或重新插拔 USB 摄像头。
8. WSL2 专属问题
摄像头无法使用：配置 WSL2 USB 设备穿透，使用 usbip 挂载硬件；
CUDA 调用异常：安装适配 WSL2 的 CUDA 驱动，保证版本与 PyTorch 匹配；
音频无法播放：额外配置 WSL2 音频服务 + 安装 mpg123。

