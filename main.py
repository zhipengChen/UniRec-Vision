import os
import cv2
import numpy as np
import time
import queue
import threading
import asyncio
import edge_tts
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ===================== 全局性能参数 =====================
RECOG_INTERVAL = 1.5
TTS_INTERVAL = 3.0
YOLO_SIZE = 320
# 分开阈值：人脸偏低，物品设更高防止误检
FACE_SIM_THRESHOLD = 0.45
OBJ_SIM_THRESHOLD = 0.80
MAX_QUEUE_SIZE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAM_URL = "http://192.168.1.6:4747/video"
# 统一单数据库目录
MAIN_DB_DIR = "unified_database"
os.makedirs(MAIN_DB_DIR, exist_ok=True)

# YOLO 人物类别ID
PERSON_CLS_ID = 0

# 全局退出标记 + 超时阈值
EXIT_FLAG = threading.Event()
JOIN_TIMEOUT = 1.5

# ===================== TTS 语音模块 =====================
VOICE = "zh-CN-XiaoyiNeural"
TEMP_AUDIO = "/tmp/sent_tts.mp3"
tts_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
last_tts_time = dict()

def tts_worker():
    while not EXIT_FLAG.is_set():
        try:
            text = tts_queue.get(timeout=0.3)
            if text is None:
                break
            now = time.time()
            if text in last_tts_time and (now - last_tts_time[text]) < TTS_INTERVAL:
                tts_queue.task_done()
                continue

            async def run_tts(t):
                comm = edge_tts.Communicate(t, VOICE)
                await comm.save(TEMP_AUDIO)
                os.system(f"mpg123 -q {TEMP_AUDIO}")

            try:
                asyncio.run(run_tts(text))
            except Exception:
                pass
            last_tts_time[text] = now
            tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            continue

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# ===================== 字体绘制模块 =====================
FONT_SIZE = 20
GLOBAL_FONT = None

def init_font():
    global GLOBAL_FONT
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simhei.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                GLOBAL_FONT = ImageFont.truetype(p, FONT_SIZE, index=0)
                return
            except Exception:
                continue
    GLOBAL_FONT = ImageFont.load_default()

init_font()

def draw_chinese(img, text, pos, color=(0,255,0)):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=GLOBAL_FONT, fill=color)
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_text_wh(text):
    dummy = Image.new("RGB", (1,1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0,0), text, font=GLOBAL_FONT)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

# ===================== MobileNetV2 物品特征提取器 =====================
mobilenet_model = None
transform = None

def init_mobilenet():
    global mobilenet_model, transform
    mobilenet_model = models.mobilenet_v2(pretrained=True)
    mobilenet_model = mobilenet_model.features
    mobilenet_model.eval()
    mobilenet_model.to(DEVICE)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

@torch.no_grad()
def get_mobilenet_feature(img_crop_bgr):
    img_rgb = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    feat_map = mobilenet_model(tensor)
    feat = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
    feat = feat.view(-1).cpu().numpy()
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat = feat / norm
    return feat

# ===================== 统一向量相似度计算 =====================
def vec_similarity(feat1, feat2):
    return np.dot(feat1, feat2)

# ===================== 统一数据库管理 =====================
# type: 0=人脸  1=物品
unified_db = dict()

def load_db():
    global unified_db
    unified_db.clear()
    count = 0
    for f in os.listdir(MAIN_DB_DIR):
        if f.endswith(".npz"):
            name = f[:-4].strip()
            try:
                data = np.load(os.path.join(MAIN_DB_DIR, f), allow_pickle=True)
                feat = data["feat"]
                feat_type = int(data["type"])
                unified_db[name] = (feat, feat_type)
                count += 1
            except Exception:
                continue
    print(f"📂 加载统一特征库: 共 {count} 条记录")

def save_feature(name, feat, feat_type):
    save_path = os.path.join(MAIN_DB_DIR, f"{name}.npz")
    np.savez(save_path, feat=feat, type=feat_type)
    unified_db[name] = (feat, feat_type)
    type_name = "人脸" if feat_type == 0 else "物品"
    print(f"✅ {type_name}注册成功: {name}")

# ===================== 全局模型加载 =====================
print(f"⏳ 加载模型，运行设备: {DEVICE}")
detector = YOLO("yolov8n.pt")
detector.fuse()

init_mobilenet()

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == "cuda" else ['CPUExecutionProvider']
session_options = None
try:
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
except ImportError:
    pass

face_app = None
try:
    face_app = FaceAnalysis(
        name="buffalo_l",
        root=".",
        providers=providers,
        session_options=session_options
    )
    face_app.prepare(ctx_id=0, det_size=(224, 224))
except Exception as e:
    if "out of memory" in str(e).lower():
        print("⚠️ 显存不足，人脸模型切换 CPU")
        providers = ['CPUExecutionProvider']
        face_app = FaceAnalysis(
            name="buffalo_l",
            root=".",
            providers=providers,
            session_options=session_options
        )
        face_app.prepare(ctx_id=0, det_size=(224, 224))

# ===================== 异步推理线程 =====================
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
last_recog_time = dict()

def inference_thread():
    while not EXIT_FLAG.is_set():
        try:
            frame = frame_queue.get(timeout=0.3)
            now = time.time()
            h, w = frame.shape[:2]

            res = detector(frame, imgsz=YOLO_SIZE, verbose=False, device=DEVICE)
            boxes = res[0].boxes.xyxy.cpu().numpy() if len(res[0].boxes) > 0 else []
            cls_ids = res[0].boxes.cls.cpu().numpy() if len(res[0].boxes) > 0 else []
            out = []

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                box_key = (x1, y1, x2, y2)
                cls_id = int(cls_ids[idx])

                if box_key in last_recog_time and (now - last_recog_time[box_key]) < RECOG_INTERVAL:
                    out.append((box, "Unknown", 0.0, cls_id))
                    continue

                cx1 = max(0, x1 - 5)
                cy1 = max(0, y1 - 5)
                cx2 = min(w, x2 + 5)
                cy2 = min(h, y2 + 5)
                crop = frame[cy1:cy2, cx1:cx2]

                match_name = "Unknown"
                max_sim = 0.0
                curr_type = 0 if cls_id == PERSON_CLS_ID else 1
                curr_feat = None

                # 提取特征
                if curr_type == 0:
                    faces = face_app.get(crop)
                    if len(faces) > 0:
                        raw_feat = faces[0].embedding
                        norm = np.linalg.norm(raw_feat)
                        curr_feat = raw_feat / norm if norm > 1e-8 else raw_feat
                else:
                    curr_feat = get_mobilenet_feature(crop)

                # 按类型 + 独立阈值比对
                if curr_feat is not None:
                    thresh = FACE_SIM_THRESHOLD if curr_type == 0 else OBJ_SIM_THRESHOLD
                    for name, (db_feat, db_type) in unified_db.items():
                        if db_type != curr_type:
                            continue
                        sim = vec_similarity(curr_feat, db_feat)
                        if sim > max_sim and sim > thresh:
                            max_sim = sim
                            match_name = name

                last_recog_time[box_key] = now
                out.append((box, match_name, max_sim, cls_id))

            result_queue.put(out)
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            continue

inf_thread = threading.Thread(target=inference_thread, daemon=True)
inf_thread.start()

# ===================== 主线程 =====================
def main():
    global last_recog_time
    load_db()
    cap = cv2.VideoCapture(CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    if not cap.isOpened():
        print("❌ 无法打开视频流")
        return

    print("📷 运行中 | 空格=注册(自动人脸/物品) | ESC=退出")
    last_draw_result = []

    while not EXIT_FLAG.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(CAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        try:
            last_draw_result = result_queue.get_nowait()
            result_queue.task_done()
        except queue.Empty:
            pass

        current_names = set()
        for (box, name, sim, cls_id) in last_draw_result:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({sim:.2f})"
            tw, th = get_text_wh(label)
            tx = x1 + ((x2 - x1) - tw) // 2
            ty = y1 + ((y2 - y1) - th) // 2
            draw_chinese(frame, label, (tx, ty), color)

            if name != "Unknown":
                current_names.add(name)

        for name in current_names:
            try:
                tts_queue.put_nowait(name)
            except queue.Full:
                pass

        cv2.imshow("Face & Object Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        # 空格注册
        if key == 32:
            res = detector(frame, imgsz=YOLO_SIZE, verbose=False, device=DEVICE)
            if len(res[0].boxes) == 0:
                print("⚠️ 未检测到目标，无法注册")
                continue

            box = res[0].boxes.xyxy[0].cpu().numpy()
            cls_id = int(res[0].boxes.cls[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            cx1 = max(0, x1 - 5)
            cy1 = max(0, y1 - 5)
            cx2 = min(w, x2 + 5)
            cy2 = min(h, y2 + 5)
            crop = frame[cy1:cy2, cx1:cx2]

            draw_chinese(frame, "注册中...", (50, 50), (0, 255, 255))
            cv2.imshow("Face & Object Recognition", frame)

            reg_name = input("\n👤 输入注册名称: ").strip()
            if not reg_name:
                print("⚠️ 取消注册")
                continue

            if cls_id == PERSON_CLS_ID:
                faces = face_app.get(crop)
                if len(faces) > 0:
                    raw_feat = faces[0].embedding
                    norm = np.linalg.norm(raw_feat)
                    feat = raw_feat / norm if norm > 1e-8 else raw_feat
                    save_feature(reg_name, feat, feat_type=0)
                else:
                    print("❌ 人脸特征提取失败")
            else:
                feat = get_mobilenet_feature(crop)
                save_feature(reg_name, feat, feat_type=1)

        # ESC 退出
        elif key == 27:
            EXIT_FLAG.set()
            break

    # 优雅退出
    print("\n🛑 开始退出程序...")
    cap.release()
    cv2.destroyAllWindows()

    last_recog_time.clear()
    unified_db.clear()

    for _ in range(10):
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except queue.Empty:
            break
    for _ in range(10):
        try:
            result_queue.get_nowait()
            result_queue.task_done()
        except queue.Empty:
            break

    inf_thread.join(timeout=JOIN_TIMEOUT)

    for _ in range(10):
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    tts_queue.put(None)
    tts_thread.join(timeout=JOIN_TIMEOUT)

    if os.path.exists(TEMP_AUDIO):
        try:
            os.remove(TEMP_AUDIO)
        except Exception:
            pass

    print("👋 程序已安全退出")

if __name__ == "__main__":
    main()
