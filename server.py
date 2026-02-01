from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
import asyncio
from threading import Thread

# Import our new capture logic
from audio_stream_handler import SystemAudioCapture
from lstm_handler import LSTMPredictor

# Import Model Logic
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from model import RawNet2 
from collections import OrderedDict

# Configuration
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CHUNKS_DIR = os.path.join(STATIC_DIR, "chunks")

os.makedirs(CHUNKS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- RawNet2 Model Handling ---
d_args = { 'nb_samp': 64600, 'first_conv': 1024, 'in_channels': 1, 'filts': [20, [20, 20], [20, 128]], 'gru_node': 1024, 'nb_gru_layer': 3, 'nb_fc_node': 1024, 'gru_drop': 0.5, 'do_add': True, 'do_mul': True, 'nb_classes': 2 }

class ModelHandler:
    def __init__(self, model_paths):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if isinstance(model_paths, str):
            model_paths = [model_paths]
            
        for path in model_paths:
            if os.path.exists(path):
                print(f"[Model] Attempting to load: {path}")
                if self.load_model(path):
                    break
            else:
                print(f"[Model] Path not found: {path}")
        
        if self.model is None:
             print("[Model] All load attempts failed. Using Simulation Mode.")
        
    def load_model(self, path):
        try:
            self.model = RawNet2(d_args)
            if self.device == 'cpu':
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                except TypeError:
                    checkpoint = torch.load(path, map_location='cpu')
            else:
                try:
                    checkpoint = torch.load(path, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(path)
            
            if isinstance(checkpoint, dict):
                new_state_dict = OrderedDict()
                # Check for state_dict key or just use dict
                if 'state_dict' in checkpoint:
                    print("  -> Found 'state_dict' key.")
                    checkpoint = checkpoint['state_dict']
                elif 'model' in checkpoint:
                     print("  -> Found 'model' key.")
                     checkpoint = checkpoint['model']

                for k, v in checkpoint.items():
                    name = k.replace("module.", "") if "module." in k else k
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                print("  -> Loaded full model object.")
                self.model = checkpoint

            self.model.to(self.device).eval()
            print("[Model] Success: RawNet2 loaded")
            return True
        except Exception as e:
            print(f"[Model] Load failed for {path}: {e}")
            self.model = None
            return False

    def predict(self, audio_path):
        if self.model is None:
            import random
            score_fake = random.random()
            score_real = 1.0 - score_fake
            label = "Real" if score_real > score_fake else "Fake"
            confidence = max(score_real, score_fake)
            return {"label": label, "score": confidence, "simulated": True}
            
        try:
            data, sr = sf.read(audio_path)
            # Ensure float32
            data = data.astype(np.float32)
            
            target_len = 64600
            if len(data) < target_len:
                num_repeats = int(target_len / len(data)) + 1
                data = np.tile(data, num_repeats)[:target_len]
            else:
                data = data[:target_len]
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(data_tensor)
                probs = F.softmax(output, dim=1)
                score_fake = probs[0][0].item()
                score_real = probs[0][1].item()
                label = "Real" if score_real > score_fake else "Fake"
                confidence = max(score_real, score_fake)
                return {"label": label, "score": confidence, "simulated": False}
        except Exception as e:
            print(f"[Model] Inference error: {e}")
            return {"label": "Error", "score": 0.0, "simulated": True}

MODEL_PATHS = [
    r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\PROJECTTO\final_clean_model.pth",
    r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\RawNet2_VoIP_Tuned_Ep2.pth",
    r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\RawNet2_VoIP_Tuned_Ep2_model.pth"
]
model_handler = ModelHandler(MODEL_PATHS)

# --- LSTM Predictor ---
LSTM_MODEL_PATH = r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\PROJECTTO\best_lstm.h5"
SCALER_PATH = r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\PROJECTTO\scaler.joblib"
lstm_predictor = LSTMPredictor(scaler_path=SCALER_PATH, model_path=LSTM_MODEL_PATH)


# --- Audio Capture STATE ---
capture_queue = asyncio.Queue()
capturer = None
capture_thread = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/start_capture")
async def start_capture():
    global capturer, capture_thread
    
    if capturer and capturer.is_recording:
        return {"status": "already_running"}
    
    try:
        # Get current loop for callback
        loop = asyncio.get_running_loop()
    except RuntimeError:
         # Should not happen in uvicorn
         loop = asyncio.new_event_loop()
    
    def thread_callback(filename, index):
        asyncio.run_coroutine_threadsafe(capture_queue.put((filename, index)), loop)
    
    capturer = SystemAudioCapture(static_dir=CHUNKS_DIR, callback=thread_callback)
    
    # Start thread (Windows WASAPI)
    capture_thread = Thread(target=capturer.capture_system_audio_windows)
    capture_thread.start()
    
    return {"status": "started"}

@app.get("/stop_capture")
async def stop_capture():
    global capturer
    if capturer:
        capturer.stop_recording()
    return {"status": "stopped"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Monitor connected")
    
    try:
        while True:
            # Wait for next chunk
            filename, index = await capture_queue.get()
            filepath = os.path.join(CHUNKS_DIR, filename)
            
            # 1. RawNet Prediction
            rawnet_res = model_handler.predict(filepath)
            
            # 2. LSTM Prediction
            lstm_score = lstm_predictor.predict(filepath)
            
            # 3. Combine / Log
            # LSTM score is likely probability of Fake (or Real?)
            # Usually 0=Fake, 1=Real. 
            # RawNet: Real if score_real > score_fake.
            
            # We will send BOTH to UI if needed, but for now stick to RawNet as primary source for label
            # unless user wants weighted average.
            
            print(f"File: {filename} | RawNet: {rawnet_res['label']} ({rawnet_res['score']:.2f}) | LSTM: {lstm_score}")
            
            # Pass LSTM score in data so UI can use it later if updated
            rawnet_res['lstm_score'] = float(lstm_score) if lstm_score is not None else -1
            
            # Send
            await websocket.send_json({
                "status": "new_chunk",
                "filename": filename,
                "chunk_index": index,
                "prediction": rawnet_res
            })
            
    except WebSocketDisconnect:
        print("[WebSocket] Monitor disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8007")
    uvicorn.run(app, host="127.0.0.1", port=8007)
