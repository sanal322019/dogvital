import asyncio
import time
import json
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import os

# ================= CONFIG =================
MAX_POINTS = 200

STEP_NEG_THRESHOLD = -0.25
STEP_POS_THRESHOLD = 0.24

CUTOFF_FREQ = 2.0
SAMPLING_RATE = 20
SG_WINDOW = 11
SG_POLYORDER = 2
SAMPLE_TOLERANCE = 5
VALLEY_DELAY = 1.0

# ================= FILTER =================
def lowpass_filter(data, cutoff=CUTOFF_FREQ, fs=SAMPLING_RATE, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

# ================= STATE =================
class DogState:
    def __init__(self):
        self.stretch_data = deque(maxlen=MAX_POINTS)
        self.ppg_data = deque(maxlen=MAX_POINTS)
        self.sample_count = 0

        self.step_state = 0
        self.step_count = 0

        self.valley_count = 0
        self.counted_valley_abs = set()
        self.last_valley_time = 0
        self.minute_valley_count = 0
        self.minute_valley_start = time.time()

        self.last_detected_peak = -9999
        self.minute_beat_count = 0
        self.minute_beat_start = time.time()

        self.gps_link = "GPS:No Fix"
        self.history = []
        self.history_step_start = 0
        self.latest_payload = {}

state = DogState()

# ================= CONNECTION MANAGER =================
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ================= LOAD HTML =================
def load_html():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()

# ================= FASTAPI =================
app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse(load_html())

@app.get("/health")
async def health():
    return {"status": "ok"}

# ESP32 connects here
@app.websocket("/esp")
async def esp_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 connected")
    try:
        async for message in websocket.iter_text():
            await process_line(message.strip())
    except WebSocketDisconnect:
        print("ESP32 disconnected")
    except Exception as e:
        print(f"ESP32 error: {e}")

# Browser connects here
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("Browser connected")
    try:
        if state.latest_payload:
            await websocket.send_text(json.dumps(state.latest_payload))
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)

# ================= PROCESS LINE =================
async def process_line(line: str):
    s = state

    if line.startswith("GPS:"):
        s.gps_link = line
        print(line)
        await manager.broadcast({"type": "gps", "value": line})
        return

    parts = line.split()
    if len(parts) != 3:
        return

    try:
        roll = float(parts[0])
        stretch = int(parts[1])
        ppg = int(parts[2])
    except:
        return

    if s.step_state == 0:
        if roll <= STEP_NEG_THRESHOLD:
            s.step_state = 1
    elif s.step_state == 1:
        if roll >= STEP_POS_THRESHOLD:
            s.step_count += 1
            s.step_state = 0

    s.stretch_data.append(stretch)
    s.ppg_data.append(ppg)
    s.sample_count += 1

    payload = {
        "type": "data",
        "step_count": s.step_count,
        "valley_count": s.valley_count,
        "minute_valley_count": s.minute_valley_count,
        "minute_beat_count": s.minute_beat_count,
        "gps": s.gps_link,
        "stretch_raw": [],
        "stretch_filtered": [],
        "stretch_valleys": [],
        "ppg_centered": [],
        "ppg_peaks": [],
        "history": s.history[-20:],
    }

    if len(s.stretch_data) >= SG_WINDOW:
        raw = np.array(s.stretch_data)
        lpf = lowpass_filter(raw)
        filtered = savgol_filter(lpf, SG_WINDOW, SG_POLYORDER)

        inverted = -filtered
        valleys, _ = find_peaks(inverted, distance=40, prominence=20)
        crests, _ = find_peaks(filtered, distance=40, prominence=20)

        valid_valleys = []
        for v in valleys:
            left = [c for c in crests if c < v]
            if left:
                depth = filtered[max(left)] - filtered[v]
                if depth >= 70:
                    valid_valleys.append(v)

        abs_indices = [
            s.sample_count - (MAX_POINTS - 1 - v)
            for v in valid_valleys
        ]

        if s.sample_count >= MAX_POINTS:
            for abs_idx in abs_indices:
                if not any(abs(abs_idx - c) <= SAMPLE_TOLERANCE for c in s.counted_valley_abs):
                    current_time = time.time()
                    if current_time - s.last_valley_time > VALLEY_DELAY:
                        s.valley_count += 1
                        s.minute_valley_count += 1
                        s.last_valley_time = current_time
                    s.counted_valley_abs.add(abs_idx)

        payload["stretch_raw"] = raw.tolist()
        payload["stretch_filtered"] = filtered.tolist()
        payload["stretch_valleys"] = valid_valleys

    if len(s.ppg_data) >= SG_WINDOW:
        ppg_arr = np.array(s.ppg_data)
        ppg_f = savgol_filter(ppg_arr, 11, 2)
        centered = ppg_f - np.mean(ppg_f)

        peaks, _ = find_peaks(centered, distance=25, prominence=100)

        for p in peaks:
            abs_peak = s.sample_count - (MAX_POINTS - 1 - p)
            if abs_peak - s.last_detected_peak > 30:
                s.last_detected_peak = abs_peak
                s.minute_beat_count += 1

        payload["ppg_centered"] = centered.tolist()
        payload["ppg_peaks"] = peaks.tolist()

    now = time.time()
    if now - s.minute_valley_start >= 60:
        s.history.append({
            "time": time.strftime("%H:%M"),
            "valleys": s.minute_valley_count,
            "beats": s.minute_beat_count,
            "steps": s.step_count - s.history_step_start,
        })
        s.history_step_start = s.step_count
        s.minute_valley_count = 0
        s.minute_beat_count = 0
        s.minute_valley_start = now
        payload["history"] = s.history[-20:]

    payload["valley_count"] = s.valley_count
    payload["minute_valley_count"] = s.minute_valley_count
    payload["minute_beat_count"] = s.minute_beat_count
    payload["step_count"] = s.step_count

    s.latest_payload = payload
    await manager.broadcast(payload)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
