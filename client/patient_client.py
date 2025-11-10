import socket
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog, filedialog
import json
import datetime
import os
import sys
import time
from collections import deque

# Optional heavy deps; import lazily inside methods where possible
try:
    import numpy as np
    import cv2
except Exception:
    np = None
    cv2 = None
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# Ensure model_integration is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MODEL_INTEGRATION_DIR = os.path.join(_PROJECT_ROOT, 'model_integration')
if os.path.isdir(_MODEL_INTEGRATION_DIR) and _MODEL_INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, _MODEL_INTEGRATION_DIR)

class PatientClient:
    def __init__(self):
        self.socket = None
        self.connected = False
        self.server_host = None
        self.server_port = 12345
        self.patient_name = None
        
        # Realtime sign capture state
        self.capture_running = False
        self.capture_thread = None
        self.sequence_length = 30
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.predicted_words = []
        self.last_prediction = None
        self.last_prediction_time = 0.0
        self.prediction_cooldown_sec = 1.0
        self.min_confidence = 0.80
        self.no_hands_frames = 0
        self.hands_present_frames = 0
        self.hands_absent_threshold = 30  # frames
        self.model = None
        self.actions = []
        self.mediapipe_processor = None
        self.llm_processor = None
        self.selected_model_path = None
        self.pending_auto_messages = deque()
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Patient Communication Client")
        self.root.geometry("700x600")
        self.root.configure(bg='#f0f0f0')
        # Try auto-start camera shortly after UI is ready
        try:
            self.root.after(600, self._auto_start_camera_if_possible)
        except Exception:
            pass
        
        # Layout: left side (chat/controls), right side (camera)
        main_panes = tk.PanedWindow(self.root, orient='horizontal', sashrelief='raised', bg='#f0f0f0')
        main_panes.pack(fill='both', expand=True)

        left_panel = tk.Frame(main_panes, bg='#f0f0f0')
        right_panel = tk.Frame(main_panes, bg='#f0f0f0')
        main_panes.add(left_panel, minsize=360)
        main_panes.add(right_panel, minsize=320)

        # Title
        title_label = tk.Label(left_panel, text="Patient Communication Client", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Connection frame
        connection_frame = tk.LabelFrame(left_panel, text="Connection", 
                                       font=('Arial', 12, 'bold'), bg='#f0f0f0')
        connection_frame.pack(pady=10, padx=20, fill='x')
        
        # Server IP input
        ip_frame = tk.Frame(connection_frame, bg='#f0f0f0')
        ip_frame.pack(pady=5, padx=5, fill='x')
        
        tk.Label(ip_frame, text="Doctor's IP Address:", 
                font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        
        self.ip_entry = tk.Entry(ip_frame, font=('Arial', 10), width=20)
        self.ip_entry.pack(side='left', padx=5)
        self.ip_entry.insert(0, "192.168.1.100")  # Default IP
        
        # Patient name input
        name_frame = tk.Frame(connection_frame, bg='#f0f0f0')
        name_frame.pack(pady=5, padx=5, fill='x')
        
        tk.Label(name_frame, text="Your Name:", 
                font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        
        self.name_entry = tk.Entry(name_frame, font=('Arial', 10), width=20)
        self.name_entry.pack(side='left', padx=5)
        
        # Connection buttons
        button_frame = tk.Frame(connection_frame, bg='#f0f0f0')
        button_frame.pack(pady=5)
        
        self.connect_button = tk.Button(button_frame, text="Connect", 
                                      command=self.connect_to_server, bg='#27ae60', fg='white',
                                      font=('Arial', 10), padx=15, pady=5)
        self.connect_button.pack(side='left', padx=5)
        
        self.disconnect_button = tk.Button(button_frame, text="Disconnect", 
                                         command=self.disconnect_from_server, bg='#e74c3c', fg='white',
                                         font=('Arial', 10), padx=15, pady=5, state='disabled')
        self.disconnect_button.pack(side='left', padx=5)

        # Highly visible camera controls at top
        self.top_select_model_button = tk.Button(button_frame, text="Select Model", 
                                                 command=self.select_model_file,
                                                 bg='#8e44ad', fg='white', font=('Arial', 10), padx=10, pady=5)
        self.top_select_model_button.pack(side='left', padx=5)
        
        # Status frame
        status_frame = tk.Frame(left_panel, bg='#f0f0f0')
        status_frame.pack(pady=5, padx=20, fill='x')
        
        self.status_label = tk.Label(status_frame, text="Status: Disconnected", 
                                   font=('Arial', 12), bg='#f0f0f0', fg='red')
        self.status_label.pack(side='left')
        
        # Message area
        message_frame = tk.LabelFrame(left_panel, text="Messages", 
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0')
        message_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        self.messages_text = scrolledtext.ScrolledText(message_frame, height=15, 
                                                     font=('Arial', 10), wrap='word')
        self.messages_text.pack(pady=5, padx=5, fill='both', expand=True)
        
        # Message input frame
        input_frame = tk.Frame(message_frame, bg='#f0f0f0')
        input_frame.pack(pady=5, padx=5, fill='x')
        
        self.message_entry = tk.Entry(input_frame, font=('Arial', 12), width=50)
        self.message_entry.pack(side='left', padx=5, fill='x', expand=True)
        self.message_entry.bind('<Return>', self.send_message)
        self.message_entry.config(state='disabled')
        
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message,
                                   bg='#3498db', fg='white', font=('Arial', 10), 
                                   padx=15, pady=5, state='disabled')
        self.send_button.pack(side='right', padx=5)

        # Real-time sign capture controls
        capture_frame = tk.LabelFrame(left_panel, text="Real-time Sign Capture", 
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0')
        capture_frame.pack(pady=10, padx=20, fill='x')

        self.model_label = tk.Label(capture_frame, text="Model: Not selected", 
                                    font=('Arial', 10), bg='#f0f0f0')
        self.model_label.pack(side='left', padx=5)

        self.select_model_button = tk.Button(capture_frame, text="Select Model", 
                                             command=self.select_model_file,
                                             bg='#8e44ad', fg='white', font=('Arial', 10), padx=10, pady=5)
        self.select_model_button.pack(side='left', padx=5)

        # Embedded video preview area (same screen as connection)
        self.video_frame = tk.LabelFrame(right_panel, text="Camera Preview", 
                                       font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.video_frame.pack(pady=10, padx=20, fill='both', expand=True)
        # Use a Canvas for reliable pixel sizing
        self.video_width = 640
        self.video_height = 480
        self.video_canvas = tk.Canvas(self.video_frame, width=self.video_width, height=self.video_height, bg='#000000', highlightthickness=0)
        self.video_canvas.pack(pady=5)
        self._video_tk_ref = None  # prevent GC

        # Menu bar for access via keyboard
        menubar = tk.Menu(self.root)
        capture_menu = tk.Menu(menubar, tearoff=0)
        capture_menu.add_command(label="Select Model", command=self.select_model_file)
        # Removed Start/Stop camera from menu; camera runs continuously
        menubar.add_cascade(label="Capture", menu=capture_menu)
        self.root.config(menu=menubar)
        
        # Instructions
        instructions = tk.Label(left_panel, 
                              text="Instructions: Enter the doctor's IP address and your name, then click Connect.\n"
                                   "Make sure both devices are connected to the same WiFi network.",
                              font=('Arial', 9), bg='#f0f0f0', fg='#7f8c8d', justify='center')
        instructions.pack(pady=5)
        
    def connect_to_server(self):
        server_ip = self.ip_entry.get().strip()
        patient_name = self.name_entry.get().strip()
        
        if not server_ip:
            messagebox.showerror("Error", "Please enter the doctor's IP address!")
            return
            
        if not patient_name:
            messagebox.showerror("Error", "Please enter your name!")
            return
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((server_ip, self.server_port))
            
            # Send patient information
            client_info = {
                'name': patient_name,
                'type': 'patient_info'
            }
            self.socket.send(json.dumps(client_info).encode('utf-8'))
            
            self.connected = True
            self.server_host = server_ip
            self.patient_name = patient_name
            
            # Update UI
            self.status_label.config(text=f"Status: Connected to {server_ip}", fg='green')
            self.connect_button.config(state='disabled')
            self.disconnect_button.config(state='normal')
            self.message_entry.config(state='normal')
            self.send_button.config(state='normal')
            self.ip_entry.config(state='disabled')
            self.name_entry.config(state='disabled')
            
            # Start receiving messages thread
            receive_thread = threading.Thread(target=self.receive_messages, daemon=True)
            receive_thread.start()
            
            self.log_message(f"Connected to doctor at {server_ip}")
            # Flush any queued auto messages
            try:
                while self.pending_auto_messages:
                    queued = self.pending_auto_messages.popleft()
                    self.auto_send_message(queued)
            except Exception as _:
                pass
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect to server: {str(e)}")
            if self.socket:
                self.socket.close()
                self.socket = None
                
    def disconnect_from_server(self):
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        # Update UI
        self.status_label.config(text="Status: Disconnected", fg='red')
        self.connect_button.config(state='normal')
        self.disconnect_button.config(state='disabled')
        self.message_entry.config(state='disabled')
        self.send_button.config(state='disabled')
        self.ip_entry.config(state='normal')
        self.name_entry.config(state='normal')
        
        self.log_message("Disconnected from doctor")
        
    def receive_messages(self):
        while self.connected:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                    
                message_data = json.loads(data)
                self.root.after(0, lambda msg=message_data: self.handle_incoming_message(msg))
                
            except Exception as e:
                if self.connected:
                    self.root.after(0, lambda: self.log_message(f"Error receiving message: {str(e)}"))
                    self.root.after(0, lambda: self.disconnect_from_server())
                break
                
    def handle_incoming_message(self, message_data):
        message_type = message_data.get('type', '')
        message = message_data.get('message', '')
        sender = message_data.get('sender', 'Unknown')
        
        if message_type == 'system':
            self.log_message(f"SYSTEM: {message}")
            return
        if message_type == 'doctor_voice':
            # Auto-play voice
            try:
                import base64, io
                import sounddevice as sd
                import soundfile as sf
                b64 = message_data.get('audio', '')
                if b64:
                    data = base64.b64decode(b64)
                    mem = io.BytesIO(data)
                    audio, sr = sf.read(mem, dtype='int16')
                    sd.play(audio, sr)
                    sd.wait()
                    self.log_message("Playing doctor's voice message...")
                else:
                    self.log_message("Received empty voice message")
            except Exception as e:
                self.log_message(f"Voice playback failed: {e}")
            return
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] {sender}: {message}\n")
        self.messages_text.see(tk.END)
            
    def send_message(self, event=None):
        if not self.connected:
            messagebox.showwarning("Warning", "Not connected to doctor!")
            return
            
        message = self.message_entry.get().strip()
        if not message:
            return
            
        try:
            timestamp = datetime.datetime.now().isoformat()
            message_data = {
                'type': 'patient_message',
                'message': message,
                'timestamp': timestamp,
                'sender': self.patient_name
            }
            
            self.socket.send(json.dumps(message_data).encode('utf-8'))
            
            # Display message in patient's chat
            timestamp_display = datetime.datetime.now().strftime("%H:%M:%S")
            self.messages_text.insert(tk.END, f"[{timestamp_display}] You: {message}\n")
            self.messages_text.see(tk.END)
            
            self.message_entry.delete(0, tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send message: {str(e)}")
            self.disconnect_from_server()

    def auto_send_message(self, message: str):
        """Send a message programmatically to the doctor and log it locally."""
        if not message:
            return
        if not self.connected:
            # Queue and inform user; will send when connected
            self.pending_auto_messages.append(message)
            self.log_message("Not connected. Queued message to send after connecting.")
            return
        try:
            timestamp = datetime.datetime.now().isoformat()
            message_data = {
                'type': 'patient_message',
                'message': message,
                'timestamp': timestamp,
                'sender': self.patient_name
            }
            self.socket.send(json.dumps(message_data).encode('utf-8'))
            timestamp_display = datetime.datetime.now().strftime("%H:%M:%S")
            self.messages_text.insert(tk.END, f"[{timestamp_display}] You: {message}\n")
            self.messages_text.see(tk.END)
        except Exception as e:
            self.log_message(f"Error auto-sending message: {e}")
            self.disconnect_from_server()

    # ==== Real-time sign capture integration ====
    def select_model_file(self):
        """Allow user to select a trained .h5 model file."""
        initial_dir = os.path.join(_MODEL_INTEGRATION_DIR, 'src', 'models', 'trained')
        if not os.path.isdir(initial_dir):
            initial_dir = _PROJECT_ROOT
        path = filedialog.askopenfilename(title="Select Trained Model (.h5)",
                                          initialdir=initial_dir,
                                          filetypes=(("Keras Model", "*.h5"), ("All Files", "*.*")))
        if path:
            self.selected_model_path = path
            name = os.path.basename(path)
            self.model_label.config(text=f"Model: {name}")
            self.log_message(f"Selected model: {path}")

    def lazy_import_runtime_modules(self):
        """Import heavy modules and helpers at runtime to avoid startup issues."""
        global np, cv2
        if np is None or cv2 is None:
            try:
                import numpy as _np  # noqa
                import cv2 as _cv2   # noqa
                np = _np
                cv2 = _cv2
            except Exception as e:
                self.log_message(f"Missing dependencies (opencv-python, numpy). {e}")
                return False
        if Image is None or ImageTk is None:
            try:
                from PIL import Image as _Image  # noqa
                from PIL import ImageTk as _ImageTk  # noqa
                globals()['Image'] = _Image
                globals()['ImageTk'] = _ImageTk
            except Exception as e:
                self.log_message(f"Missing dependency: Pillow (PIL). {e}")
                return False
        try:
            from model_integration.src.utils.robust_mediapipe import RobustMediaPipeProcessor, prob_viz_safe  # noqa
            from model_integration.src.utils.model_utils import SignLanguageModel  # noqa
        except Exception as e:
            self.log_message(f"Error importing model utilities: {e}")
            return False
        try:
            from model_integration.gemini_integration import create_llm_processor  # noqa
            # Cache instance
            if self.llm_processor is None:
                self.llm_processor = create_llm_processor()
        except Exception as e:
            # LLM is optional; we'll fallback if not available
            self.log_message(f"LLM not available, will fallback to basic enhancement. {e}")
        return True

    def ensure_model_loaded(self):
        """Load the trained model and actions from metadata if not yet loaded."""
        try:
            from model_integration.src.utils.model_utils import SignLanguageModel
        except Exception as e:
            self.log_message(f"Cannot import SignLanguageModel: {e}")
            return False

        if self.model is not None and self.actions:
            return True

        model_path = self.selected_model_path
        if not model_path:
            # Try to find any .h5 in default folder
            default_dir = os.path.join(_MODEL_INTEGRATION_DIR, 'src', 'models', 'trained')
            if os.path.isdir(default_dir):
                candidates = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith('.h5')]
                if candidates:
                    model_path = sorted(candidates)[0]
                    self.selected_model_path = model_path
                    self.model_label.config(text=f"Model: {os.path.basename(model_path)}")
        if not model_path:
            self.log_message("No model selected. Please select a .h5 model.")
            return False

        try:
            # Create a temporary model wrapper just to load and read metadata
            tmp = SignLanguageModel(actions=['Action_1'])
            self.model = tmp.load_model(model_path)
            self.actions = tmp.actions
            if not self.actions:
                self.actions = [f'Action_{i+1}' for i in range(self.model.output_shape[1])]
            # Suppress UI log for actions to keep screen clean
            try:
                print(f"Model loaded. Actions: {', '.join(self.actions)}")
            except Exception:
                pass
            return True
        except Exception as e:
            self.log_message(f"Failed to load model: {e}")
            return False

    def start_sign_capture(self):
        if self.capture_running:
            return
        if not self.lazy_import_runtime_modules():
            return
        # Note: We don't require the model at start. We'll lazy-load when a hand is detected.
        # Initialize mediapipe
        try:
            from model_integration.src.utils.robust_mediapipe import RobustMediaPipeProcessor
            self.mediapipe_processor = RobustMediaPipeProcessor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception as e:
            self.log_message(f"Failed to initialize MediaPipe: {e}")
            return

        self.capture_running = True
        self.predicted_words = []
        self.last_prediction = None
        self.last_prediction_time = 0.0
        self.sequence_buffer.clear()
        self.no_hands_frames = 0
        self.hands_present_frames = 0

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.log_message("Started sign capture. Show your hands to begin.")

    def stop_sign_capture(self):
        if not self.capture_running:
            return
        self.capture_running = False
        try:
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
        except Exception:
            pass
        # Keep camera running; no buttons to toggle
        # Immediately restart capture loop to allow next gesture without pressing Start
        try:
            self.start_sign_capture()
        except Exception:
            pass
        self.log_message("Sign capture cycled for next gesture.")

    def _predict_from_sequence(self, sequence_array):
        try:
            preds = self.model.predict(np.expand_dims(sequence_array, axis=0))[0]
            return preds
        except Exception as e:
            self.log_message(f"Prediction error: {e}")
            return None

    def _generate_sentence(self, raw_words: str) -> str:
        text = raw_words.strip()
        if not text:
            return ""
        # Prefer LLM if available
        try:
            if self.llm_processor is not None:
                return self.llm_processor.process_sign_language_output(text)
        except Exception as e:
            self.log_message(f"LLM processing failed, using fallback. {e}")
        # Fallback: simple enhancement
        try:
            from model_integration.gemini_integration import SignLanguageLLMProcessor
            return SignLanguageLLMProcessor._basic_enhance_sentence(self=None, words=text)  # type: ignore
        except Exception:
            # Minimal fallback
            words = text.split()
            if len(words) >= 6:
                return text
            return f"I am communicating: {' '.join(words)} and need help"

    def _update_video_label(self, bgr_frame):
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Resize to fixed canvas size
            img = img.resize((self.video_width, self.video_height))
            imgtk = ImageTk.PhotoImage(image=img)
            self._video_tk_ref = imgtk
            self.video_canvas.create_image(0, 0, image=imgtk, anchor='nw')
        except Exception:
            pass

    def _capture_loop(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: self.log_message("Cannot access camera."))
                self.stop_sign_capture()
                return
            # Colors placeholder for viz util
            colors = None
            from model_integration.src.utils.robust_mediapipe import prob_viz_safe
            prev_hands_visible = False
            while self.capture_running:
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = self.mediapipe_processor.detect_safe(frame)
                # Draw only hand landmarks onto image
                try:
                    image = self.mediapipe_processor.draw_landmarks_safe(image, results)
                except Exception:
                    pass
                # Check hand visibility
                left = getattr(results, 'left_hand_landmarks', None) if results is not None else None
                right = getattr(results, 'right_hand_landmarks', None) if results is not None else None
                hands_visible = left is not None or right is not None
                if hands_visible:
                    self.hands_present_frames += 1
                    self.no_hands_frames = 0
                    # On first visible hand after absence, start a new session buffer
                    if not prev_hands_visible:
                        self.sequence_buffer.clear()
                        self.predicted_words = []
                        self.last_prediction = None
                        self.last_prediction_time = 0.0
                        # Lazy-load model on first need
                        if self.model is None:
                            # try auto
                            self.ensure_model_loaded()
                            # If still None, prompt user to select model (non-blocking)
                            if self.model is None:
                                self.root.after(0, self.select_model_file)
                else:
                    self.no_hands_frames += 1
                # Extract keypoints
                if hands_visible:
                    keypoints = self.mediapipe_processor.extract_keypoints_safe(results)
                    self.sequence_buffer.append(keypoints)
                prediction = None
                preds = None
                if len(self.sequence_buffer) == self.sequence_length and hands_visible and self.model is not None:
                    preds = self._predict_from_sequence(np.array(self.sequence_buffer))
                    if preds is not None and len(preds) == len(self.actions):
                        max_idx = int(np.argmax(preds))
                        max_conf = float(preds[max_idx])
                        if max_conf >= self.min_confidence:
                            prediction = self.actions[max_idx]
                # Debounce predictions
                now = time.time()
                if prediction:
                    if (prediction != self.last_prediction) or (now - self.last_prediction_time >= self.prediction_cooldown_sec):
                        self.predicted_words.append(prediction)
                        self.last_prediction = prediction
                        self.last_prediction_time = now
                # Visualize inside Tkinter (no predicted word overlay)
                try:
                    self.root.after(0, lambda f=image.copy(): self._update_video_label(f))
                except Exception:
                    pass
                if (self.no_hands_frames >= self.hands_absent_threshold) and len(self.predicted_words) > 0:
                    # Stop and generate sentence
                    break
                prev_hands_visible = hands_visible
            # Cleanup
            try:
                cap.release()
            except Exception:
                pass
            # Generate and send
            raw = ' '.join(self.predicted_words)
            if raw:
                sentence = self._generate_sentence(raw)
                self.root.after(0, lambda: self.log_message(f"Predicted sentence: {sentence}"))
                self.root.after(0, lambda s=sentence: self.auto_send_message(s))
            # Reset UI/state
            self.root.after(0, self.stop_sign_capture)
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Capture loop error: {e}"))
            self.root.after(0, self.stop_sign_capture)

    # ==== Auto start helpers ====
    def _auto_start_camera_if_possible(self):
        """On app open, attempt to select a model and start camera automatically."""
        # If already running, do nothing
        if self.capture_running:
            return
        # Ensure deps
        if not self.lazy_import_runtime_modules():
            return
        # Pick first model if none selected
        if not self.selected_model_path:
            default_dir = os.path.join(_MODEL_INTEGRATION_DIR, 'src', 'models', 'trained')
            if os.path.isdir(default_dir):
                candidates = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith('.h5')]
                if candidates:
                    self.selected_model_path = sorted(candidates)[0]
                    self.model_label.config(text=f"Model: {os.path.basename(self.selected_model_path)}")
        # Start capture
        try:
            self.start_sign_capture()
        except Exception as e:
            self.log_message(f"Auto-start camera failed: {e}")
            
    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.messages_text.see(tk.END)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PatientClient()
    app.run()
