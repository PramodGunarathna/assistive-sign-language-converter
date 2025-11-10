import json
import base64
import cv2
import numpy as np
import threading
import time
import sys
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import logging

# Add the parent directory to path to import the transformer model
sys.path.append('..')
from transformerencoder_prediction import VideoTransformerEncoder, SignLanguageRecognitionModel
import torch
from transformers import T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

# Import Gemini integration
try:
    from gemini_integration import create_llm_processor
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARNING: Gemini integration not available")
    GEMINI_AVAILABLE = False

# Import configuration
try:
    from config import (
        MODEL_PATHS, NETWORK_CONFIG, VIDEO_CONFIG, HAND_DETECTION_CONFIG,
        MODEL_CONFIG, GUI_CONFIG, LOGGING_CONFIG, PERFORMANCE_CONFIG,
        get_model_path, validate_config
    )
except ImportError:
    # Fallback configuration if config.py is not found
    print("WARNING: Config module not found, using default configuration")
    
    # Default configuration
    MODEL_PATHS = {
        'i3d_model': 'flow_imagenet.pt',
        'recognition_model': 'best_model2_with_Transformers_28000.pth',
        'tokenizer': 'saved_tokenizer_T5Decoder'
    }
    
    NETWORK_CONFIG = {
        'default_host': '0.0.0.0',
        'default_port': 3456,
        'socket_timeout': 0.01,
        'max_connections': 5
    }
    
    VIDEO_CONFIG = {
        'frame_size': 224,
        'min_recording_frames': 30,
        'hand_detection_threshold': 5,
        'fps': 30,
        'compression_quality': 80
    }
    
    HAND_DETECTION_CONFIG = {
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5
    }
    
    MODEL_CONFIG = {
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 2048,
        'dropout': 0.1,
        'max_length': 50,
        'num_beams': 4
    }
    
    GUI_CONFIG = {
        'server_window_size': '1000x700',
        'client_window_size': '1600x1000',
        'webcam_display_size': (1200, 800),
        'font_family': 'Arial',
        'title_font_size': 24,
        'normal_font_size': 12,
        'small_font_size': 10
    }
    
    LOGGING_CONFIG = {
        'log_level': 'INFO',
        'log_file': 'sign_language_system.log',
        'max_log_size': 10 * 1024 * 1024,
        'backup_count': 5
    }
    
    PERFORMANCE_CONFIG = {
        'enable_gpu': True,
        'batch_size': 1,
        'num_workers': 0,
        'pin_memory': True,
        'enable_optimization': True
    }
    
    def get_model_path(model_name):
        """Get the full path to a model file."""
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = MODEL_PATHS[model_name]
        if not os.path.isabs(model_path):
            # Get the project root directory (parent of client_server_system)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, model_path)
        
        return model_path
    
    def validate_config():
        """Validate that all required model files exist."""
        missing_files = []
        
        for model_name in MODEL_PATHS:
            model_path = get_model_path(model_name)
            if not os.path.exists(model_path):
                missing_files.append(f"{model_name}: {model_path}")
        
        if missing_files:
            print("ERROR: Missing model files:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        print("SUCCESS: All model files found")
        return True

# Socket imports
import socket

# Speech recognition imports
try:
    import speech_recognition as sr
except ImportError:
    print("Speech recognition not found. Please install: pip install SpeechRecognition")
    sys.exit(1)

class SignLanguageServer:
    """Server for receiving video data and processing sign language predictions."""
    
    def __init__(self, host=None, port=None, frame_size=None):
        # Use configuration defaults if not provided
        self.host = host or NETWORK_CONFIG['default_host']
        self.port = port or NETWORK_CONFIG['default_port']
        self.frame_size = frame_size or VIDEO_CONFIG['frame_size']
        
        # Setup device
        if PERFORMANCE_CONFIG['enable_gpu'] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("INFO: Using GPU acceleration")
        else:
            self.device = torch.device("cpu")
            print("INFO: Using CPU processing")
        
        # Model paths from configuration
        self.i3d_model_path = get_model_path('i3d_model')
        self.recognition_model_path = get_model_path('recognition_model')
        self.tokenizer_path = get_model_path('tokenizer')
        
        # Setup logging
        self.setup_logging()
        
        # Validate configuration
        if not validate_config():
            raise RuntimeError("Configuration validation failed. Please check model files.")
        
        # Initialize LLM processor before GUI setup
        self.llm_processor = None
        if GEMINI_AVAILABLE:
            try:
                self.llm_processor = create_llm_processor()
                print("SUCCESS: Gemini LLM processor initialized")
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini LLM processor: {e}")
                self.llm_processor = None
        else:
            print("WARNING: Gemini LLM processor not available")
        
        # GUI setup first
        self.setup_gui()
        
        # Load models after GUI is ready
        self.load_models()
        
        # Update LLM status after GUI and models are loaded
        self.update_llm_status()
        
        # Socket server
        self.server = None
        self.clients = []
        
        # Voice recording (simple start/stop approach)
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.recording = False
        self.audio_data = None
        
        # Video processing (no display needed)
        self.video_processing_enabled = True
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOGGING_CONFIG['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load I3D model, recognition model, and tokenizer."""
        print("INFO: STEP 1: Starting model loading process...")
        self.update_status("Step 1: Loading models...", "blue")
        
        try:
            # Load I3D model
            print(f"INFO: STEP 2: Loading I3D model from: {self.i3d_model_path}...")
            self.update_status("Step 2: Loading I3D model...", "blue")
            
            # Check if I3D model file exists
            import os
            if not os.path.exists(self.i3d_model_path):
                error_msg = f"ERROR: TROUBLESHOOT: I3D model file not found at {self.i3d_model_path}"
                print(error_msg)
                print("💡 SOLUTION: Check if flow_imagenet.pt exists in the correct location")
                print("💡 ALTERNATIVE: Update model path in config or move file to correct location")
                self.update_status("I3D model file missing", "red")
                raise FileNotFoundError(f"I3D model file not found: {self.i3d_model_path}")
            
            print(f"SUCCESS: I3D model file found: {os.path.getsize(self.i3d_model_path)} bytes")
            
            from pytorch_i3d import InceptionI3d
            self.i3d_model = InceptionI3d(num_classes=400, in_channels=2)
            self.i3d_model.load_state_dict(torch.load(self.i3d_model_path, map_location=self.device, weights_only=True))
            self.i3d_model.to(self.device)
            self.i3d_model.eval()
            print("SUCCESS: STEP 2: I3D model loaded successfully")
            self.update_status("Step 2: I3D model OK", "green")
            
            # Load tokenizer
            print(f"INFO: STEP 3: Loading T5 tokenizer from: {self.tokenizer_path}...")
            self.update_status("Step 3: Loading tokenizer...", "blue")
            
            if not os.path.exists(self.tokenizer_path):
                error_msg = f"ERROR: TROUBLESHOOT: Tokenizer directory not found at {self.tokenizer_path}"
                print(error_msg)
                print("💡 SOLUTION: Check if saved_tokenizer_T5Decoder folder exists with all required files")
                print("💡 REQUIRED FILES: tokenizer_config.json, spiece.model, added_tokens.json, special_tokens_map.json")
                self.update_status("Tokenizer directory missing", "red")
                raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_path}")
            
            print(f"SUCCESS: Tokenizer directory found")
            self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_path)
            print("SUCCESS: STEP 3: Tokenizer loaded successfully")
            self.update_status("Step 3: Tokenizer OK", "green")
            
            # Load recognition model
            print(f"INFO: STEP 4: Loading recognition model from: {self.recognition_model_path}...")
            self.update_status("Step 4: Loading recognition model...", "blue")
            
            if not os.path.exists(self.recognition_model_path):
                error_msg = f"ERROR: TROUBLESHOOT: Recognition model file not found at {self.recognition_model_path}"
                print(error_msg)
                print("💡 SOLUTION: Check if best_model2_with_Transformers_28000.pth exists in the correct location")
                self.update_status("Recognition model file missing", "red")
                raise FileNotFoundError(f"Recognition model file not found: {self.recognition_model_path}")
            
            print(f"SUCCESS: Recognition model file found: {os.path.getsize(self.recognition_model_path)} bytes")
            
            encoder = VideoTransformerEncoder(
                input_dim=1024, 
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                num_layers=MODEL_CONFIG['num_layers'],
                num_heads=MODEL_CONFIG['num_heads'],
                ff_dim=MODEL_CONFIG['ff_dim'],
                dropout=MODEL_CONFIG['dropout']
            )
            self.recognition_model = SignLanguageRecognitionModel(
                encoder, 
                t5_model_name='t5-small', 
                hidden_dim=MODEL_CONFIG['hidden_dim']
            )
            
            state_dict = torch.load(self.recognition_model_path, map_location=self.device, weights_only=True)
            self.recognition_model.load_state_dict(state_dict)
            self.recognition_model.to(self.device)
            self.recognition_model.eval()
            print("SUCCESS: STEP 4: Recognition model loaded successfully")
            self.update_status("Step 4: Recognition model OK", "green")
            
            print("SUCCESS: STEP 5: All models loaded successfully!")
            self.update_status("All models loaded successfully", "green")
            
        except Exception as e:
            error_msg = f"ERROR: TROUBLESHOOT: Error loading models: {e}"
            print(error_msg)
            print(f"ERROR: ERROR TYPE: {type(e).__name__}")
            print("💡 TROUBLESHOOTING STEPS:")
            print("   1. Check if all model files exist in correct locations")
            print("   2. Verify file permissions (read access)")
            print("   3. Check available disk space")
            print("   4. Verify PyTorch installation and version")
            print("   5. Check CUDA/CPU compatibility")
            print("   6. Verify model file integrity (not corrupted)")
            import traceback
            traceback.print_exc()
            self.update_status("Model loading failed", "red")
            raise e
    
    def setup_gui(self):
        """Setup the GUI for doctor's communication interface."""
        self.root = tk.Tk()
        self.root.title("Doctor Communication Interface - Sign Language Recognition")
        self.root.geometry(GUI_CONFIG['server_window_size'])
        self.root.configure(bg="#f0f0f0")
        
        # Main title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="MEDICAL: DOCTOR COMMUNICATION INTERFACE", 
                              font=("Arial", 24, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(expand=True)
        
        # Status section
        status_frame = tk.Frame(self.root, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="Status: Waiting for patient connection...", 
                                   font=("Arial", 14, "bold"), fg="#e74c3c", bg="#ecf0f1")
        self.status_label.pack(pady=5)
        
        # LLM status label
        llm_status = "SUCCESS: LLM Enhanced" if self.llm_processor else "WARNING: LLM Not Available"
        llm_color = "#27ae60" if self.llm_processor else "#f39c12"
        self.llm_status_label = tk.Label(status_frame, text=f"LLM Status: {llm_status}", 
                                       font=("Arial", 10, "bold"), fg=llm_color, bg="#ecf0f1")
        self.llm_status_label.pack(pady=2)
        
        # Main content frame with messages and voice
        main_content_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Patient messages (70% width)
        left_frame = tk.Frame(main_content_frame, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Patient sign language messages section
        patient_frame = tk.LabelFrame(left_frame, text="INFO: Patient's Sign Language Translation", 
                                    font=("Arial", 16, "bold"), fg="#2c3e50", bg="#f0f0f0")
        patient_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.prediction_text = scrolledtext.ScrolledText(patient_frame, wrap=tk.WORD, 
                                                        font=("Arial", 18), width=80, height=15, 
                                                        bg="#e8f4f8", fg="#2c3e50", relief=tk.SUNKEN, bd=2)
        self.prediction_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Right side - Voice communication (30% width)
        right_frame = tk.Frame(main_content_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Doctor's voice section
        voice_frame = tk.LabelFrame(right_frame, text="VOICE: Doctor's Voice Messages", 
                                  font=("Arial", 14, "bold"), fg="#2c3e50", bg="#f0f0f0")
        voice_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.voice_text = scrolledtext.ScrolledText(voice_frame, wrap=tk.WORD, 
                                                   font=("Arial", 12), width=40, height=8, 
                                                   bg="#fff3cd", fg="#856404", relief=tk.SUNKEN, bd=2)
        self.voice_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Voice control section
        voice_control_frame = tk.LabelFrame(right_frame, text="VOICE: Voice Controls", 
                                          font=("Arial", 12, "bold"), fg="#2c3e50", bg="#f0f0f0")
        voice_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Voice status and controls
        self.voice_status_label = tk.Label(voice_control_frame, text="Voice: Ready", 
                                         font=("Arial", 10), fg="#27ae60", bg="#f0f0f0")
        self.voice_status_label.pack(pady=5)
        
        # Voice input text box
        self.voice_input_frame = tk.Frame(voice_control_frame, bg="#f0f0f0")
        self.voice_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(self.voice_input_frame, text="Quick Message:", 
                font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W)
        
        self.quick_message_entry = tk.Entry(self.voice_input_frame, font=("Arial", 10), width=35)
        self.quick_message_entry.pack(fill=tk.X, pady=2)
        self.quick_message_entry.bind('<Return>', self.send_quick_message)
        
        # Send quick message button
        send_quick_btn = tk.Button(self.voice_input_frame, text="Send Quick Message", 
                                  command=self.send_quick_message,
                                  font=("Arial", 10, "bold"), bg="#3498db", fg="white",
                                  relief=tk.RAISED, bd=2, padx=10, pady=2)
        send_quick_btn.pack(pady=2)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Left side buttons
        left_buttons = tk.Frame(button_frame, bg="#f0f0f0")
        left_buttons.pack(side=tk.LEFT)
        
        clear_button = tk.Button(left_buttons, text="CLEAR: Clear All", command=self.clear_text,
                               font=("Arial", 14, "bold"), bg="#95a5a6", fg="white", 
                               relief=tk.RAISED, bd=3, padx=20, pady=10)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Center - Voice Recording Buttons
        center_buttons = tk.Frame(button_frame, bg="#f0f0f0")
        center_buttons.pack(side=tk.LEFT, expand=True)
            
        # Start Recording Button
        self.start_record_button = tk.Button(center_buttons, text="VOICE: START RECORDING", command=self.start_voice_recording,
                                        font=("Arial", 16, "bold"), bg="#27ae60", fg="white",
                                        relief=tk.RAISED, bd=5, padx=30, pady=10)
        self.start_record_button.pack(pady=5)
        
        # Stop Recording Button
        self.stop_record_button = tk.Button(center_buttons, text="VOICE: STOP & TRANSCRIBE", command=self.stop_voice_recording,
                                        font=("Arial", 16, "bold"), bg="#e74c3c", fg="white",
                                        relief=tk.RAISED, bd=5, padx=30, pady=10)
        self.stop_record_button.pack(pady=5)
        self.stop_record_button.config(state=tk.DISABLED)  # Disabled initially
            
        
        
        # Right side buttons
        right_buttons = tk.Frame(button_frame, bg="#f0f0f0")
        right_buttons.pack(side=tk.RIGHT)
        
        quit_button = tk.Button(right_buttons, text="ERROR: Exit", command=self.quit_server,
                              font=("Arial", 14, "bold"), bg="#34495e", fg="white",
                              relief=tk.RAISED, bd=3, padx=20, pady=10)
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # Server info
        info_frame = tk.Frame(self.root, bg="#34495e", height=30)
        info_frame.pack(fill=tk.X, side=tk.BOTTOM)
        info_frame.pack_propagate(False)
        
        info_label = tk.Label(info_frame, text=f"Server running on {self.host}:{self.port} | Ready for patient connection", 
                            font=("Arial", 10), fg="white", bg="#34495e")
        info_label.pack(expand=True)
    
    def clear_text(self):
        """Clear the prediction text."""
        self.prediction_text.delete("1.0", tk.END)
        self.voice_text.delete("1.0", tk.END)
    
    def start_voice_recording(self):
        """Start voice recording."""
        try:
            # Initialize microphone if not already done
            if self.microphone is None:
                self.microphone = sr.Microphone()
                print("VOICE: Microphone initialized")
            
            # Configure recognizer for recording
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 1.0
            
            # Start recording
            self.recording = True
            self.start_record_button.config(state=tk.DISABLED)
            self.stop_record_button.config(state=tk.NORMAL)
            
            self.voice_status_label.config(text="Voice: Recording...", fg="#e74c3c")
            self.voice_text.insert(tk.END, "VOICE: Recording started. Speak now...\n")
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
        except Exception as e:
            error_msg = f"ERROR: Error starting recording: {e}"
            print(error_msg)
            self.voice_text.insert(tk.END, f"{error_msg}\n")
            self.voice_status_label.config(text="Voice: Error", fg="#e74c3c")
    
    def stop_voice_recording(self):
        """Stop voice recording and transcribe."""
        try:
            # Stop recording
            self.recording = False
            
            # Update button states
            self.start_record_button.config(state=tk.NORMAL)
            self.stop_record_button.config(state=tk.DISABLED)
            
            self.voice_status_label.config(text="Voice: Transcribing...", fg="#f39c12")
            self.voice_text.insert(tk.END, "VOICE: Recording stopped. Transcribing...\n")
            
            # Transcribe the recorded audio
            if self.audio_data:
                self.transcribe_audio()
            else:
                self.voice_text.insert(tk.END, "VOICE: No audio recorded.\n")
                self.voice_status_label.config(text="Voice: No audio", fg="#95a5a6")
                
        except Exception as e:
            error_msg = f"ERROR: Error stopping recording: {e}"
            print(error_msg)
            self.voice_text.insert(tk.END, f"{error_msg}\n")
            self.voice_status_label.config(text="Voice: Error", fg="#e74c3c")
    
    def record_audio(self):
        """Record audio until stopped."""
        try:
            print("VOICE: Starting audio recording...")
            
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("VOICE: Recording...")
                
                # Record audio while recording is active
                while self.recording:
                    try:
                        # Listen for audio with a longer timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                        
                        if self.recording:  # Check again in case it was stopped
                            if self.audio_data is None:
                                self.audio_data = audio
                            else:
                                # Combine audio data (simple approach)
                                # Note: This is a simplified approach for demo purposes
                                self.audio_data = audio
                                
                        print("VOICE: Audio chunk recorded")
                        
                    except sr.WaitTimeoutError:
                        # No audio in this chunk, continue
                        continue
                    except Exception as e:
                        print(f"VOICE: Recording error: {e}")
                        continue
            
            print("VOICE: Recording completed")
            
        except Exception as e:
            error_msg = f"ERROR: Recording failed: {e}"
            print(error_msg)
            self.root.after(0, self.update_voice_display, f"{error_msg}\n")
    
    def transcribe_audio(self):
        """Transcribe the recorded audio."""
        try:
            print("VOICE: Starting transcription...")
            
            if self.audio_data is None:
                self.voice_text.insert(tk.END, "VOICE: No audio to transcribe.\n")
                self.voice_status_label.config(text="Voice: No audio", fg="#95a5a6")
                return
            
            # Recognize speech using Google
            text = self.recognizer.recognize_google(self.audio_data)
            
            print(f"VOICE: Transcribed text: '{text}'")
            
            if text and text.strip():
                # Clean the text
                cleaned_text = text.strip().capitalize()
                
                # Add punctuation if missing
                if cleaned_text and cleaned_text[-1] not in '.!?':
                    cleaned_text += '.'
                
                timestamp = time.strftime("%H:%M:%S")
                voice_message = f"[{timestamp}] DOCTOR: {cleaned_text}\n"
                
                # Update GUI
                self.voice_text.insert(tk.END, voice_message)
                self.voice_text.see(tk.END)
                
                # Send to clients (same as quick message)
                self.send_voice_to_clients(cleaned_text)
                
                # Update status
                self.voice_status_label.config(text=f"Voice: Sent '{cleaned_text[:25]}...'", fg="#27ae60")
                
                print(f"VOICE: Message transcribed and sent: '{cleaned_text}'")
                
            else:
                self.voice_text.insert(tk.END, "VOICE: No speech detected in recording.\n")
                self.voice_status_label.config(text="Voice: No speech", fg="#95a5a6")
            
            # Clear audio data for next recording
            self.audio_data = None
            
        except sr.UnknownValueError:
            error_msg = "VOICE: Could not understand the recorded audio."
            print(error_msg)
            self.voice_text.insert(tk.END, f"{error_msg}\n")
            self.voice_status_label.config(text="Voice: Could not understand", fg="#e74c3c")
            self.audio_data = None
            
        except sr.RequestError as e:
            error_msg = f"VOICE: API error during transcription: {e}"
            print(error_msg)
            self.voice_text.insert(tk.END, f"{error_msg}\n")
            self.voice_status_label.config(text="Voice: API error", fg="#e74c3c")
            self.audio_data = None
            
        except Exception as e:
            error_msg = f"ERROR: Transcription failed: {e}"
            print(error_msg)
            self.voice_text.insert(tk.END, f"{error_msg}\n")
            self.voice_status_label.config(text="Voice: Error", fg="#e74c3c")
            self.audio_data = None
    
    def update_voice_display(self, message):
        """Update voice text display (called from main thread)."""
        self.voice_text.insert(tk.END, message)
        self.voice_text.see(tk.END)
    
    def send_voice_to_clients(self, text):
        """Send voice text to all connected clients."""
        if not self.clients:
            print("WARNING: No clients connected to send voice message to")
            return
            
        voice_data = {
            'type': 'voice_text',
            'text': text,
            'timestamp': time.time()
        }
        
        message = json.dumps(voice_data) + '\n'  # Add newline delimiter
        print(f"SENDING: Sending voice message to {len(self.clients)} client(s): {text}")
        
        # Send to all connected clients
        for client in self.clients[:]:  # Copy list to avoid modification during iteration
            try:
                client.send(message.encode('utf-8'))
                print(f"SUCCESS: Voice message sent to client successfully")
            except Exception as e:
                print(f"ERROR: Failed to send voice message to client: {e}")
                # Remove disconnected client
                self.clients.remove(client)
    
    def send_translation_to_clients(self, translation):
        """Send translation to all connected clients."""
        if not self.clients:
            print("WARNING: No clients connected to send translation to")
            return
            
        translation_data = {
            'type': 'translation',
            'text': translation,
            'timestamp': time.time()
        }
        
        message = json.dumps(translation_data) + '\n'  # Add newline delimiter
        print(f"SENDING: Sending translation to {len(self.clients)} client(s): {translation}")
        
        # Send to all connected clients
        for client in self.clients[:]:  # Copy list to avoid modification during iteration
            try:
                client.send(message.encode('utf-8'))
                print(f"SUCCESS: Translation sent to client successfully")
            except Exception as e:
                print(f"ERROR: Failed to send translation to client: {e}")
                # Remove disconnected client
                self.clients.remove(client)
    
    def send_quick_message(self, event=None):
        """Send a quick text message to clients."""
        message = self.quick_message_entry.get().strip()
        if message:
            self.send_voice_to_clients(message)
            self.voice_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Doctor: {message}\n")
            self.voice_text.see(tk.END)
            self.quick_message_entry.delete(0, tk.END)
    
    
    
    def process_video_for_translation(self, frames):
        """Process video frames for sign language translation."""
        try:
            print(f"INFO: STEP 1: Processing {len(frames)} frames for translation...")
            self.root.after(0, self.update_status, "Step 1: Processing frames...", "blue")
            
            # Check if models are loaded
            print("INFO: STEP 2: Checking model availability...")
            self.root.after(0, self.update_status, "Step 2: Checking models...", "blue")
            
            if self.i3d_model is None:
                error_msg = "ERROR: TROUBLESHOOT: I3D model not loaded - check model path and file existence"
                print(error_msg)
                print("💡 SOLUTION: Verify flow_imagenet.pt exists in the correct location")
                self.add_prediction(error_msg)
                self.root.after(0, self.update_status, "I3D model missing", "red")
                return None
                
            if self.recognition_model is None:
                error_msg = "ERROR: TROUBLESHOOT: Recognition model not loaded - check model path and file existence"
                print(error_msg)
                print("💡 SOLUTION: Verify best_model2_with_Transformers_28000.pth exists in the correct location")
                self.add_prediction(error_msg)
                self.root.after(0, self.update_status, "Recognition model missing", "red")
                return None
                
            if self.tokenizer is None:
                error_msg = "ERROR: TROUBLESHOOT: Tokenizer not loaded - check tokenizer path and file existence"
                print(error_msg)
                print("💡 SOLUTION: Verify saved_tokenizer_T5Decoder folder exists with all required files")
                self.add_prediction(error_msg)
                self.root.after(0, self.update_status, "Tokenizer missing", "red")
                return None
            
            print("SUCCESS: STEP 2: All models loaded successfully")
            self.root.after(0, self.update_status, "Step 2: Models OK", "green")
            
            # Extract features and predict
            print("INFO: STEP 3: Extracting I3D features from frames...")
            self.root.after(0, self.update_status, "Step 3: Extracting features...", "blue")
            
            features = self.extract_features_from_frames(frames)
            if features is not None:
                print(f"SUCCESS: STEP 3: Features extracted successfully - shape: {features.shape}")
                self.root.after(0, self.update_status, "Step 3: Features extracted", "green")
                
                print("INFO: STEP 4: Generating translation prediction...")
                self.root.after(0, self.update_status, "Step 4: Generating prediction...", "blue")
                
                prediction = self.predict_from_frames(features)
                print(f"SUCCESS: STEP 4: Prediction generated: '{prediction}'")
                self.root.after(0, self.update_status, "Step 4: Prediction complete", "green")
                
                print("INFO: STEP 5: Displaying translation on server...")
                self.root.after(0, self.update_status, "Step 5: Displaying translation...", "blue")
                
                # Display translation on server
                self.add_prediction(f" {prediction}")
                print("SUCCESS: STEP 5: Translation displayed on server")
                self.root.after(0, self.update_status, "Step 5: Server display OK", "green")
                
                print("INFO: STEP 6: Sending translation to client...")
                self.root.after(0, self.update_status, "Step 6: Sending to client...", "blue")
                
                # Send translation to client as well
                self.send_translation_to_clients(prediction)
                print("SUCCESS: STEP 6: Translation sent to client")
                self.root.after(0, self.update_status, "Step 6: Client message sent", "green")
                
                print("SUCCESS: TRANSLATION COMPLETE: All steps successful!")
                self.root.after(0, self.update_status, "Translation completed successfully", "green")
                
                return prediction
            else:
                error_msg = "ERROR: TROUBLESHOOT: Failed to extract features from video frames"
                print(error_msg)
                print("💡 POSSIBLE CAUSES:")
                print("   - Video frames are corrupted or empty")
                print("   - I3D model is not working properly")
                print("   - Frame format is incorrect (should be BGR)")
                print("   - Not enough frames for processing")
                self.add_prediction(error_msg)
                self.root.after(0, self.update_status, "Feature extraction failed", "red")
                return None
                
        except Exception as e:
            error_msg = f"ERROR: TROUBLESHOOT: Error processing video: {str(e)}"
            print(error_msg)
            print(f"ERROR: ERROR TYPE: {type(e).__name__}")
            print("💡 TROUBLESHOOTING STEPS:")
            print("   1. Check if all model files exist")
            print("   2. Verify CUDA/CPU compatibility")
            print("   3. Check available memory")
            print("   4. Verify frame format and size")
            print("   5. Check model loading logs")
            import traceback
            traceback.print_exc()
            self.add_prediction(error_msg)
            self.root.after(0, self.update_status, "Translation error", "red")
            return None
    
    
    
    def quit_server(self):
        """Quit the server application."""
        if messagebox.askokcancel("Quit", "Do you want to quit the server?"):
            self.root.quit()
            self.root.destroy()
    
    def update_status(self, message, color="black"):
        """Update the status label."""
        try:
            if hasattr(self, 'status_label') and self.status_label is not None:
                self.status_label.config(text=f"Status: {message}", fg=color)
                self.root.update()
            else:
                print(f"Status: {message}")  # Fallback to console if GUI not ready
        except Exception as e:
            print(f"Error updating status: {e}")
            print(f"Status: {message}")  # Fallback to console
    
    def update_llm_status(self):
        """Update the LLM status label."""
        try:
            if hasattr(self, 'llm_status_label') and self.llm_status_label is not None:
                llm_status = "SUCCESS: LLM Enhanced" if self.llm_processor else "WARNING: LLM Not Available"
                llm_color = "#27ae60" if self.llm_processor else "#f39c12"
                self.llm_status_label.config(text=f"LLM Status: {llm_status}", fg=llm_color)
                self.root.update()
        except Exception as e:
            print(f"Error updating LLM status: {e}")
    
    def add_prediction(self, prediction, timestamp=None):
        """Add a new prediction to the display."""
        try:
            # Clear previous messages and show only the latest
            if hasattr(self, 'prediction_text') and self.prediction_text is not None:
                self.prediction_text.delete("1.0", tk.END)  # Clear all text
                self.prediction_text.insert(tk.END, f" {prediction}\n")
                self.prediction_text.see(tk.END)
                self.root.update()
            else:
                print(f"Prediction: {prediction}")  # Fallback to console
        except Exception as e:
            print(f"Error adding prediction: {e}")
            print(f"Prediction: {prediction}")
    
    def calculate_flow_frames(self, frames):
        """Calculate optical flow between consecutive frames."""
        if len(frames) < 2:
            return None
            
        flow_frames = []
        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            curr_frame = frames[i + 1]
            
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = np.nan_to_num(flow)
            
            # Normalize to [-1, 1]
            flow_min, flow_max = flow.min(), flow.max()
            if flow_max > flow_min:
                flow = (flow - flow_min) / (flow_max - flow_min) * 2 - 1
            
            # Stack x and y components
            flow = np.stack([flow[..., 0], flow[..., 1]], axis=0)
            flow_frames.append(flow)
        
        return np.stack(flow_frames)
    
    def extract_features_from_frames(self, frames):
        """Extract I3D features from recorded frames."""
        flow_frames = self.calculate_flow_frames(frames)
        if flow_frames is None:
            return None
            
        with torch.no_grad():
            x = torch.from_numpy(flow_frames).float()
            x = x.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, 2, T, H, W)
            features = self.i3d_model.extract_features(x).squeeze().cpu().numpy()
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            else:
                features = features.T  # Transpose to (T', D)
                
        return features
    
    def predict_from_frames(self, features):
        """Generate prediction from extracted features."""
        if features is None:
            return "No prediction available"
            
        with torch.no_grad():
            video_features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            
            # Encode features
            encoder_output_features = self.recognition_model.encoder(video_features)
            projected_features = self.recognition_model.projection(encoder_output_features)
            encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=projected_features)
            
            # Generate prediction
            generated_ids = self.recognition_model.decoder.generate(
                encoder_outputs=encoder_outputs_for_t5,
                max_length=MODEL_CONFIG['max_length'],
                num_beams=MODEL_CONFIG['num_beams'],
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            raw_prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Enhance prediction using LLM if available
            if self.llm_processor and raw_prediction.strip():
                print(f"INFO: Enhancing prediction with LLM: '{raw_prediction}'")
                try:
                    enhanced_prediction = self.llm_processor.process_sign_language_output(raw_prediction)
                    print(f"SUCCESS: LLM enhanced prediction: '{enhanced_prediction}'")
                    return enhanced_prediction
                except Exception as e:
                    print(f"WARNING: LLM enhancement failed: {e}, using raw prediction")
                    return raw_prediction
            else:
                print(f"INFO: Using raw prediction (no LLM): '{raw_prediction}'")
                return raw_prediction
    
    def process_video_data(self, video_data):
        """Process received video data and generate translation."""
        try:
            print(f"INFO: Starting video processing for translation...")
            frames = []
            
            # Decode base64 frames
            print(f"RECEIVING: Decoding {len(video_data['frames'])} frames...")
            for i, frame_base64 in enumerate(video_data['frames']):
                frame_data = base64.b64decode(frame_base64)
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                
                # Decode as color frame and convert to grayscale for processing
                color_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if color_frame is not None:
                    # Convert to grayscale for processing
                    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_frame)
                
                if i % 10 == 0:  # Progress indicator
                    print(f"RECEIVING: Decoded {i+1}/{len(video_data['frames'])} frames")
            
            print(f"SUCCESS: Successfully decoded {len(frames)} frames")
            
            if len(frames) < 2:
                error_msg = f"ERROR: Not enough frames for processing (got {len(frames)})"
                print(error_msg)
                self.add_prediction(error_msg)
                return
            
            print(f"INFO: Processing {len(frames)} frames for sign language translation...")
            self.root.after(0, self.update_status, f"Processing {len(frames)} frames for translation...", "blue")
            
            # Process video for translation
            prediction = self.process_video_for_translation(frames)
            
            if prediction:
                print(f"SUCCESS: Translation completed: {prediction}")
            else:
                print("ERROR: Translation failed")
                
        except Exception as e:
            error_msg = f"ERROR: Error processing video: {str(e)}"
            print(error_msg)
            self.add_prediction(error_msg)
            self.root.after(0, self.update_status, "Processing error", "red")
            import traceback
            traceback.print_exc()
    
    def handle_client_message(self, ws, message):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            
            if data['type'] == 'video_data':
                print(f"Received video data with {len(data['frames'])} frames")
                self.update_status(f"Received video from client ({len(data['frames'])} frames)", "blue")
                
                # Process video in a separate thread to avoid blocking
                processing_thread = threading.Thread(
                    target=self.process_video_data, 
                    args=(data,)
                )
                processing_thread.daemon = True
                processing_thread.start()
                
            elif data['type'] == 'ping':
                # Respond to ping with pong
                pong_data = {'type': 'pong', 'timestamp': time.time()}
                ws.send(json.dumps(pong_data))
                
        except Exception as e:
            print(f"Error handling client message: {e}")
    
    def on_message(self, ws, message):
        """WebSocket message handler."""
        self.handle_client_message(ws, message)
    
    def on_open(self, ws):
        """WebSocket connection opened."""
        print("SUCCESS: Client connected!")
        self.clients.append(ws)
        self.update_status("Client connected", "green")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        print("ERROR: Client disconnected!")
        if ws in self.clients:
            self.clients.remove(ws)
        self.update_status("Client disconnected", "orange")
    
    def on_error(self, ws, error):
        """WebSocket error handler."""
        print(f"WebSocket error: {error}")
        self.update_status(f"Error: {error}", "red")
    
    def start_server(self):
        """Start the socket server."""
        
        print(f"Starting server on {self.host}:{self.port}")
        self.update_status(f"Starting server on {self.host}:{self.port}", "blue")
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=self.run_socket_server)
        server_thread.daemon = True
        server_thread.start()
        
        print("SUCCESS: Server started successfully!")
        self.update_status("Server running - waiting for client", "green")
        
        # Start GUI main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nSTOPPED: Server interrupted by user")
        finally:
            self.stop_server()
    
    def run_socket_server(self):
        """Run the socket server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        server_socket.bind((self.host, self.port))
        
        server_socket.listen(1)
        
        print(f"Server listening on {self.host}:{self.port}")
        
        while True:
            try:
                conn, addr = server_socket.accept()
                print(f"SUCCESS: Client connected from {addr}")
                self.update_status("Client connected", "green")
                
                # Add client to list
                self.clients.append(conn)
                
                # Handle client in a separate thread
                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                print(f"Server error: {e}")
                break
        
        server_socket.close()
    
    def handle_client(self, conn, addr):
        """Handle client connection."""
        try:
            # Set socket timeout to prevent blocking
            conn.settimeout(1.0)  # 1 second timeout
            
            # Keep connection alive and handle multiple messages
            while True:
                try:
                    # Receive data
                    data = b""
                    while True:
                        try:
                            chunk = conn.recv(4096)
                            if not chunk:
                                print(f"ERROR: Client {addr} closed connection")
                                return
                            data += chunk
                            
                            # Try to decode as complete JSON message
                            try:
                                message = data.decode('utf-8')
                                video_data = json.loads(message)
                                
                                if video_data.get('type') == 'video_data':
                                    print(f"Received video data with {len(video_data['frames'])} frames")
                                    # Update status in main thread
                                    self.root.after(0, self.update_status, f"Received video from client ({len(video_data['frames'])} frames)", "blue")
                                    
                                    # Process video in a separate thread
                                    processing_thread = threading.Thread(
                                        target=self.process_video_data, 
                                        args=(video_data,)
                                    )
                                    processing_thread.daemon = True
                                    processing_thread.start()
                                    
                                    # Reset data buffer for next message
                                    data = b""
                                    break
                                    
                                elif video_data.get('type') == 'ping':
                                    # Respond to ping with pong
                                    pong_data = {
                                        'type': 'pong',
                                        'timestamp': time.time()
                                    }
                                    pong_message = json.dumps(pong_data) + '\n'
                                    try:
                                        conn.send(pong_message.encode('utf-8'))
                                        print(f"NETWORK: Responded to ping from {addr}")
                                    except:
                                        pass
                                    
                                    # Reset data buffer for next message
                                    data = b""
                                    break
                                    
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # Continue receiving more data
                                continue
                                
                        except socket.timeout:
                            # No data received, continue to next iteration
                            break
                            
                except socket.timeout:
                    # No activity, continue listening
                    continue
                    
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            # Remove client from list before closing
            if conn in self.clients:
                self.clients.remove(conn)
            conn.close()
            print(f"ERROR: Client {addr} disconnected")
            # Update status in main thread
            self.root.after(0, self.update_status, "Client disconnected", "orange")
    
    def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.shutdown()
            print("Server stopped.")


def get_local_ip():
    """Get the local IP address."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    print("=" * 60)
    print("SERVER:  DOCTOR COMMUNICATION INTERFACE")
    print("=" * 60)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"Your local IP address: {local_ip}")
    
    # Get server configuration
    host_input = input(f"Enter server IP (press Enter for 0.0.0.0, or 'local' for {local_ip}): ").strip()
    if not host_input:
        host = "0.0.0.0"
    elif host_input.lower() == "local":
        host = local_ip
    else:
        host = host_input
    
    port_input = input("Enter server port (press Enter for 3456): ").strip()
    if not port_input:
        port = 3456
    else:
        port = int(port_input)
    
    print(f"Starting server on {host}:{port}")
    print("-" * 60)
    
    try:
        # Create and start server
        server = SignLanguageServer(host=host, port=port)
        server.start_server()
    except KeyboardInterrupt:
        print("\nSTOPPED: Server interrupted by user")
    except Exception as e:
        print(f"ERROR: Server error: {e}")
    finally:
        print("INFO: Server closed. Goodbye!")


if __name__ == "__main__":
    main()
