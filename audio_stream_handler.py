import os
import shutil
import soundfile as sf
import numpy as np
import pyaudiowpatch as pyaudio  # For Windows loopback
import soundcard as sc  # Cross-platform alternative
import wave
from threading import Thread, Event
import queue
import time # Added missing import

class SystemAudioCapture:
    def __init__(self, static_dir="static/chunks", callback=None):
        self.static_dir = static_dir
        self.chunk_duration = 3.0  # seconds
        self.sample_rate = 16000
        self.channels = 1
        self.chunks_available = []
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.callback = callback # Function to call when a chunk is ready
        
        os.makedirs(self.static_dir, exist_ok=True)

    def reset(self):
        """Clear previous chunks"""
        if os.path.exists(self.static_dir):
            shutil.rmtree(self.static_dir)
        os.makedirs(self.static_dir, exist_ok=True)
        self.chunks_available = []
        print("[SystemAudio] Reset complete.")

    # ==================== METHOD 2: Using PyAudio with WASAPI Loopback (Windows) ====================
    def capture_system_audio_windows(self, duration=None):
        """
        Capture system audio on Windows using WASAPI loopback
        Requires: pip install pyaudiowpatch
        """
        print("[SystemAudio] Starting Windows WASAPI loopback capture...")
        self.reset()
        
        chunk_index = 0
            
        try:
            p = pyaudio.PyAudio()
            
            # Find loopback device
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            
            if not default_speakers["isLoopbackDevice"]:
                # Find loopback device
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
            
            print(f"[SystemAudio] Using: {default_speakers['name']}")
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=default_speakers["maxInputChannels"],
                rate=int(default_speakers["defaultSampleRate"]),
                frames_per_buffer=1024,
                input=True,
                input_device_index=default_speakers["index"]
            )
            
            print("[SystemAudio] Recording system audio...")
            
            chunk_samples = int(self.chunk_duration * default_speakers["defaultSampleRate"])
            frames = []
            
            start_time = time.time()
            
            self.is_recording = True
            
            while self.is_recording:
                # Record chunk
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
                
                # Check if we have enough for 3 seconds
                total_frames = len(frames) * 1024
                if total_frames >= chunk_samples:
                    # Convert to numpy array
                    audio_data = b''.join(frames)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Convert to float and normalize
                    audio_array = audio_array.astype(np.float32) / 32768.0
                    
                    # Convert to mono if stereo
                    if default_speakers["maxInputChannels"] > 1:
                        audio_array = audio_array.reshape(-1, default_speakers["maxInputChannels"])
                        audio_array = audio_array[:, 0]
                    
                    # Resample if needed
                    if int(default_speakers["defaultSampleRate"]) != self.sample_rate:
                        import librosa
                        # Librosa resample expects (channels, samples) or (samples)
                        audio_array = librosa.resample(
                            audio_array,
                            orig_sr=int(default_speakers["defaultSampleRate"]),
                            target_sr=self.sample_rate
                        )
                    
                    # Save chunk
                    filename = f"live_chunk_{chunk_index}.wav"
                    out_path = os.path.join(self.static_dir, filename)
                    
                    sf.write(out_path, audio_array, self.sample_rate)
                    self.chunks_available.append(filename)
                    
                    print(f"[SystemAudio] Saved {filename}")
                    
                    # Trigger callback
                    if self.callback:
                        self.callback(filename, chunk_index)
                    
                    chunk_index += 1
                    frames = []
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"[SystemAudio] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_recording = False
            print(f"[SystemAudio] Capture stopped. {len(self.chunks_available)} chunks saved.")
    
    def stop_recording(self):
        self.is_recording = False
