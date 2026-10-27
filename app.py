import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import io
import cv2
import uuid
import httpx
import numpy as np
import librosa
from PIL import Image
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
import modal

# KONFIGURASI URL BACKEND
BACKEND_CALLBACK_VISION_URL = "https://capstone-diponeocare-production.up.railway.app/api/ai/results/vision"
BACKEND_CALLBACK_AUDIO_URL = "https://capstone-diponeocare-production.up.railway.app/api/ai/results/audio"

TEST_MAC_ADDRESS = "3C:0F:02:DB:FB:5C"

AUDIO_CLASSES = ["hungry", "pain", "discomfort", "dont-cry"]
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 64
MAX_LEN = 130


class AudioCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


image = (
    modal.Image.debian_slim()
    .env({"YOLO_CONFIG_DIR": "/tmp"})
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "ultralytics",
        "opencv-python-headless",
        "pillow",
        "python-multipart",
        "fastapi",
        "httpx",
        "torch",
        "librosa",
        "soundfile",
        "pydub"
    )
    .add_local_file("best_aktivitas.pt", remote_path="/root/best_aktivitas.pt")
    .add_local_file("best_bantal.pt", remote_path="/root/best_bantal.pt")
    .add_local_file("best_tubuh_muka_bayi.pt", remote_path="/root/best_tubuh_muka_bayi.pt")
    .add_local_file("best_mainan.pt", remote_path="/root/best_mainan.pt")
    .add_local_file("best_audio_classifier_menangis.pth", remote_path="/root/best_audio_classifier_menangis.pth")
)

app = modal.App("smartvision-activity-api")
web_app = FastAPI()

pending_crying_requests = modal.Dict.from_name("pending-crying-requests", create_if_missing=True)


@app.cls(image=image)
class EnsembleModelRunner:
    @modal.enter()
    def load_models(self):
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_aktivitas = YOLO("/root/best_aktivitas.pt")
        self.model_bantal = YOLO("/root/best_bantal.pt")
        self.model_tubuh_muka = YOLO("/root/best_tubuh_muka_bayi.pt")
        self.model_mainan = YOLO("/root/best_mainan.pt")

        self.model_audio = AudioCNN(num_classes=4).to(self.device)
        self.model_audio.load_state_dict(
            torch.load("/root/best_audio_classifier_menangis.pth", map_location=self.device)
        )
        self.model_audio.eval()

        self.executor = ThreadPoolExecutor(max_workers=4)

    @modal.exit()
    def shutdown_executor(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    @modal.method()
    def predict_visual(self, image_bytes: bytes):
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        f_akt = self.executor.submit(self.model_aktivitas, frame, conf=0.50)
        f_bnt = self.executor.submit(self.model_bantal, frame, conf=0.50)
        f_tbh = self.executor.submit(self.model_tubuh_muka, frame, conf=0.50)
        f_mnn = self.executor.submit(self.model_mainan, frame, conf=0.50)

        res_akt = f_akt.result()
        res_bnt = f_bnt.result()
        res_tbh = f_tbh.result()
        res_mnn = f_mnn.result()

        sleeping, awake, visual_crying = False, False, False
        akt_conf = 0.0

        if len(res_akt[0].boxes) > 0:
            box = res_akt[0].boxes[0]
            label = str(res_akt[0].names[int(box.cls[0].cpu().numpy())]).strip().lower()
            akt_conf = float(box.conf[0].cpu().numpy())

            if label in ["sleeping", "tidur"]:
                sleeping = True
            elif label in ["awake", "bangun"]:
                awake = True
            elif label in ["crying", "menangis"]:
                visual_crying = True
                awake = True

        if not sleeping and not awake:
            awake = True
            if akt_conf == 0.0:
                akt_conf = 0.85

        baby_body_detected = False
        baby_face_detected = False

        for box in res_tbh[0].boxes:
            cls_name = str(res_tbh[0].names[int(box.cls[0].cpu().numpy())]).strip().lower()
            if cls_name in ["baby-body", "tubuh-bayi", "tubuh_bayi"]:
                baby_body_detected = True
            elif cls_name in ["baby-face", "muka-bayi", "muka_bayi"]:
                baby_face_detected = True

        detected_anomaly_type = None

        if len(res_bnt[0].boxes) > 0:
            cls_bnt = str(res_bnt[0].names[int(res_bnt[0].boxes[0].cls[0].cpu().numpy())]).strip().lower()
            if "guling" in cls_bnt:
                detected_anomaly_type = "guling"
            else:
                detected_anomaly_type = "bantal"
        elif len(res_mnn[0].boxes) > 0:
            detected_anomaly_type = "mainan"

        anomaly_detected = False
        anomaly_conf = 0.0

        if detected_anomaly_type and baby_body_detected and not baby_face_detected:
            anomaly_detected = True
            anomaly_conf = 0.90
        elif (len(res_bnt[0].boxes) + len(res_mnn[0].boxes)) > 2 and detected_anomaly_type:
            anomaly_detected = True
            anomaly_conf = 0.85

        final_anomaly_type = detected_anomaly_type if anomaly_detected else None

        return {
            "sleeping": sleeping,
            "awake": awake,
            "visualCrying": visual_crying,
            "confidence": round(float(akt_conf), 2),
            "anomaly": {
                "detected": anomaly_detected,
                "type": final_anomaly_type,
                "confidence": round(float(anomaly_conf), 2)
            }
        }

    @modal.method()
    def predict_audio(self, audio_bytes: bytes):
        import soundfile as sf
        from pydub import AudioSegment

        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError("Data bytes audio terlalu kecil atau kosong.")

        y = None
        sr = SAMPLE_RATE

        try:
            audio_buf = io.BytesIO(audio_bytes)
            audio_buf.seek(0)
            
            seg = AudioSegment.from_file(audio_buf)
            seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1)
            
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            
            if seg.sample_width == 2:
                samples /= 32768.0
            elif seg.sample_width == 4:
                samples /= 2147483648.0
            elif seg.sample_width == 1:
                samples = (samples - 128.0) / 128.0
                
            y = samples
        except Exception as e_pydub:
            try:
                audio_buf = io.BytesIO(audio_bytes)
                audio_buf.seek(0)
                y, sr = sf.read(audio_buf)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                if sr != SAMPLE_RATE:
                    y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            except Exception as e_sf:
                raise ValueError(f"Gagal membaca audio via FFmpeg ({str(e_pydub)}) maupun Soundfile ({str(e_sf)})")

        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)

        if mel_spec_norm.shape[1] < MAX_LEN:
            pad = MAX_LEN - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad)), mode='constant')
        else:
            mel_spec_norm = mel_spec_norm[:, :MAX_LEN]

        inp_tensor = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model_audio(inp_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        category = AUDIO_CLASSES[pred_idx.item()]
        score = round(float(conf.cpu().numpy()[0]), 2)
        is_crying = category in ["hungry", "pain", "discomfort"]

        return {
            "isCrying": is_crying,
            "category": category if is_crying else None,
            "confidence": score
        }
async def send_callback(payload: dict, callback_url: str):
    try:
        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(callback_url, json=payload, headers=headers)
            print(f"[CALLBACK] Sent to {callback_url} | Target MAC: {payload.get('macAddress')} | Status: {res.status_code}")
            if res.status_code >= 400:
                print(f"[CALLBACK ERROR DETAILS]: {res.text}")
    except Exception as e:
        print(f"[CALLBACK ERROR]: {str(e)}")


@web_app.post("/predict/visual")
async def predict_visual(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    macAddress: Optional[str] = Form(TEST_MAC_ADDRESS),
    timestamp: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File gambar kosong.")

        runner = EnsembleModelRunner()
        vision_res = await runner.predict_visual.remote.aio(contents)

        current_timestamp = timestamp if timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        capture_id = f"capture-{uuid.uuid4().hex[:6]}"

        visual_payload = {
            "macAddress": str(macAddress),
            "captureId": str(capture_id),
            "timestamp": str(current_timestamp),
            "activity": {
                "sleeping": bool(vision_res["sleeping"]),
                "awake": bool(vision_res["awake"]),
                "confidence": float(vision_res["confidence"])
            },
            "visualCrying": {
                "detected": bool(vision_res["visualCrying"]),
                "confidence": float(vision_res["confidence"]) if vision_res["visualCrying"] else 0.0
            },
            "anomaly": {
                "detected": bool(vision_res["anomaly"]["detected"]),
                "type": vision_res["anomaly"]["type"],
                "confidence": float(vision_res["anomaly"]["confidence"])
            }
        }

        if not vision_res["visualCrying"]:
            background_tasks.add_task(send_callback, visual_payload, BACKEND_CALLBACK_VISION_URL)
            return {
                "status": "success",
                "message": "Prediksi visual selesai. Hasil dikirim ke backend via callback.",
                "data": visual_payload
            }

        pending_crying_requests[macAddress] = {
            "captureId": str(capture_id),
            "timestamp": str(current_timestamp),
            "visualConfidence": float(vision_res["confidence"]),
            "anomaly": visual_payload["anomaly"]
        }

        return {
            "status": "crying_buffered",
            "message": "Visual terdeteksi crying. Menunggu pencocokan dengan aliran audio terbaru dari backend.",
            "data": visual_payload
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses visual: {str(e)}")


@web_app.post("/predict/audio")
async def predict_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    macAddress: Optional[str] = Form(TEST_MAC_ADDRESS),
    timestamp: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File audio kosong.")

        if macAddress not in pending_crying_requests:
            return {
                "status": "ignored",
                "message": "Audio diterima, namun tidak ada status visual menangis yang menggantung untuk MAC ini."
            }

        saved_data = pending_crying_requests.pop(macAddress)
        cap_id = saved_data["captureId"]
        saved_timestamp = timestamp if timestamp else saved_data["timestamp"]
        visual_confidence = float(saved_data.get("visualConfidence", 0.0))
        saved_anomaly = saved_data["anomaly"]

        runner = EnsembleModelRunner()
        audio_res = await runner.predict_audio.remote.aio(contents)

        is_crying = bool(audio_res["isCrying"])

        crying_cls = audio_res["category"] if is_crying else None

        crying_cls_conf = float(audio_res["confidence"]) if is_crying else 0.0

        results_array = [
            {
                "captureId": str(cap_id),
                "timestamp": str(saved_timestamp),
                "isCrying": is_crying,
                "confidence": visual_confidence,
                "cryingClass": crying_cls,
                "cryingClassConfidence": crying_cls_conf
            }
            for _ in range(5)
        ]

        audio_payload = {
            "macAddress": str(macAddress),
            "results": results_array
        }

        background_tasks.add_task(send_callback, audio_payload, BACKEND_CALLBACK_AUDIO_URL)

        if is_crying:
            confirmed_visual_payload = {
                "macAddress": str(macAddress),
                "captureId": str(cap_id),
                "timestamp": str(saved_timestamp),
                "activity": {
                    "sleeping": False,
                    "awake": True,
                    "confidence": visual_confidence
                },
                "visualCrying": {
                    "detected": True,
                    "confidence": visual_confidence
                },
                "anomaly": saved_anomaly
            }
            background_tasks.add_task(send_callback, confirmed_visual_payload, BACKEND_CALLBACK_VISION_URL)

        else:
            corrected_visual_payload = {
                "macAddress": str(macAddress),
                "captureId": str(cap_id),
                "timestamp": str(saved_timestamp),
                "activity": {
                    "sleeping": False,
                    "awake": True,
                    "confidence": 0.85
                },
                "visualCrying": {
                    "detected": False,
                    "confidence": 0.0
                },
                "anomaly": saved_anomaly
            }
            background_tasks.add_task(send_callback, corrected_visual_payload, BACKEND_CALLBACK_VISION_URL)

        return {
            "status": "success",
            "message": "Pencocokan audio dan visual berhasil. Hasil validasi dikirimkan ke backend.",
            "audioData": audio_payload
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses audio: {str(e)}")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app