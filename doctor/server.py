import socket
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import datetime
import base64
import io

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None

# Optional speech recognition for voice-to-text
try:
    import speech_recognition as sr
except Exception:
    sr = None

# Text-to-speech for reading patient messages
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

from typing import Dict, List

class DoctorServer:
    def __init__(self):
        self.host = '0.0.0.0'  # Listen on all interfaces
        self.port = 12345
        self.server_socket = None
        self.clients: Dict[str, socket.socket] = {}
        self.client_names: Dict[str, str] = {}
        self.running = False
        
        # Initialize text-to-speech engine
        self.tts_engine = None
        self.tts_enabled = True  # Enable TTS by default
        self.tts_queue = []  # Queue for TTS messages
        self.tts_lock = threading.Lock()  # Lock for thread-safe queue access
        self.tts_processing = False  # Flag to track if TTS is currently speaking
        self.tts_voice_settings = {}  # Store voice settings for reuse
        self.tts_engine_initialized = False  # Track if engine has been warmed up
        
        if TTS_AVAILABLE:
            try:
                # Test initialization and get voice settings
                test_engine = pyttsx3.init()
                self.tts_voice_settings['rate'] = 150
                self.tts_voice_settings['volume'] = 0.9
                # Try to set a better voice (if available)
                voices = test_engine.getProperty('voices')
                if voices:
                    # Prefer female voice if available, otherwise use first available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_voice_settings['voice_id'] = voice.id
                            break
                test_engine.stop()  # Clean up test engine
            except Exception as e:
                print(f"Warning: Could not initialize TTS engine: {e}")
                self.tts_voice_settings = {}
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Doctor Communication Server")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(self.root, text="Doctor Communication Server", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Server status frame
        status_frame = tk.Frame(self.root, bg='#f0f0f0')
        status_frame.pack(pady=10, padx=20, fill='x')
        
        self.status_label = tk.Label(status_frame, text="Status: Stopped", 
                                   font=('Arial', 12), bg='#f0f0f0', fg='red')
        self.status_label.pack(side='left')
        
        # Server info
        self.info_label = tk.Label(status_frame, text="", 
                                 font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        self.info_label.pack(side='right')
        
        # Control buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="Start Server", 
                                    command=self.start_server, bg='#27ae60', fg='white',
                                    font=('Arial', 12), padx=20, pady=5)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop Server", 
                                   command=self.stop_server, bg='#e74c3c', fg='white',
                                   font=('Arial', 12), padx=20, pady=5, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Voice message controls
        voice_frame = tk.Frame(self.root, bg='#f0f0f0')
        voice_frame.pack(pady=5)
        self.record_button = tk.Button(voice_frame, text="Record Voice (5s)",
                                       command=self.record_and_send_voice,
                                       bg='#8e44ad', fg='white', font=('Arial', 10), padx=15, pady=5,
                                       state='normal')
        self.record_button.pack(side='left', padx=5)

        # Voice recognition controls
        self.vr_status_label = tk.Label(voice_frame, text="Voice-to-Text: Stopped",
                                        font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        self.vr_status_label.pack(side='left', padx=10)
        self.vr_toggle_button = tk.Button(voice_frame, text="Start Voice Recognition",
                                          command=self.toggle_voice_recognition,
                                          bg='#2c3e50', fg='white', font=('Arial', 10), padx=15, pady=5)
        self.vr_toggle_button.pack(side='left', padx=5)
        
        # Text-to-Speech toggle
        self.tts_toggle_button = tk.Button(voice_frame, text="🔊 TTS: ON" if self.tts_enabled else "🔇 TTS: OFF",
                                          command=self.toggle_tts,
                                          bg='#27ae60' if self.tts_enabled else '#95a5a6', 
                                          fg='white', font=('Arial', 10), padx=15, pady=5)
        self.tts_toggle_button.pack(side='left', padx=5)
        
        # Connected clients frame
        clients_frame = tk.LabelFrame(self.root, text="Connected Patients", 
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0')
        clients_frame.pack(pady=10, padx=20, fill='x')
        
        self.clients_listbox = tk.Listbox(clients_frame, height=4, font=('Arial', 10))
        self.clients_listbox.pack(pady=5, padx=5, fill='x')
        
        # Message area
        message_frame = tk.LabelFrame(self.root, text="Messages", 
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
        
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message,
                                   bg='#3498db', fg='white', font=('Arial', 10), 
                                   padx=15, pady=5, state='disabled')
        self.send_button.pack(side='right', padx=5)

    def record_and_send_voice(self, duration_sec: int = 5, samplerate: int = 16000):
        if not self.clients:
            messagebox.showwarning("Warning", "No patients connected!")
            return
        if sd is None or sf is None:
            messagebox.showerror("Missing dependency", "Please install sounddevice and soundfile: pip install sounddevice soundfile")
            return
        try:
            self.log_message("Recording voice message...")
            audio = sd.rec(int(duration_sec * samplerate), samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()
            memfile = io.BytesIO()
            sf.write(memfile, audio, samplerate, format='WAV', subtype='PCM_16')
            wav_bytes = memfile.getvalue()
            b64 = base64.b64encode(wav_bytes).decode('ascii')
            payload = {
                'type': 'doctor_voice',
                'timestamp': datetime.datetime.now().isoformat(),
                'sender': 'Doctor',
                'message': 'voice',
                'metadata': {
                    'mime': 'audio/wav',
                    'samplerate': samplerate
                },
                'audio': b64
            }
            message_json = json.dumps(payload)
            disconnected = []
            for client_id, client_socket in self.clients.items():
                try:
                    client_socket.send(message_json.encode('utf-8'))
                except:
                    disconnected.append(client_id)
            for cid in disconnected:
                if cid in self.clients:
                    del self.clients[cid]
                if cid in self.client_names:
                    del self.client_names[cid]
            self.update_clients_list()
            self.log_message("Voice message sent.")
        except Exception as e:
            messagebox.showerror("Error", f"Voice recording/sending failed: {e}")

    # ===== Voice Recognition (continuous) =====
    def toggle_voice_recognition(self):
        if not self.clients:
            messagebox.showwarning("Warning", "No patients connected!")
            # We still allow starting recognition; remove return if desired
        if sr is None:
            messagebox.showerror("Missing dependency", "Please install SpeechRecognition and PyAudio: pip install SpeechRecognition PyAudio")
            return
        # Lazy state attributes
        if not hasattr(self, 'vr_running'):
            self.vr_running = False
            self.vr_thread = None
        if self.vr_running:
            self.vr_running = False
            self.vr_toggle_button.config(text="Start Voice Recognition")
            self.vr_status_label.config(text="Voice-to-Text: Stopping...")
        else:
            self.vr_running = True
            self.vr_toggle_button.config(text="Stop Voice Recognition")
            self.vr_status_label.config(text="Voice-to-Text: Listening...")
            self.vr_thread = threading.Thread(target=self._voice_recognition_loop, daemon=True)
            self.vr_thread.start()

    def _voice_recognition_loop(self):
        try:
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = True
            # Reduce ambient noise baseline
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.8)
                except Exception:
                    pass
            while self.vr_running and self.running:
                with sr.Microphone() as source:
                    try:
                        audio = recognizer.listen(source, phrase_time_limit=6)
                    except Exception as e:
                        self.root.after(0, lambda: self.log_message(f"Voice listen error: {e}"))
                        continue
                # Recognize speech
                text = None
                try:
                    text = recognizer.recognize_google(audio, language='en-US')
                except sr.UnknownValueError:
                    # No intelligible speech
                    continue
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"Voice recognition error: {e}"))
                    continue
                if text:
                    self.root.after(0, lambda t=text: self._broadcast_doctor_text(t))
        finally:
            self.root.after(0, lambda: self.vr_status_label.config(text="Voice-to-Text: Stopped"))

    def _broadcast_doctor_text(self, message: str):
        if not message:
            return
        # Display locally
        timestamp_display = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp_display}] Doctor (voice): {message}\n")
        self.messages_text.see(tk.END)
        # Send to clients
        payload = {
            'type': 'doctor_message',
            'message': message,
            'timestamp': datetime.datetime.now().isoformat(),
            'sender': 'Doctor'
        }
        message_json = json.dumps(payload)
        disconnected = []
        for client_id, client_socket in self.clients.items():
            try:
                client_socket.send(message_json.encode('utf-8'))
            except:
                disconnected.append(client_id)
        for cid in disconnected:
            if cid in self.clients:
                del self.clients[cid]
            if cid in self.client_names:
                del self.client_names[cid]
        self.update_clients_list()
        
    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            self.status_label.config(text="Status: Running", fg='green')
            self.info_label.config(text=f"Listening on {self.get_local_ip()}:{self.port}")
            
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.send_button.config(state='normal')
            
            # Start server thread
            server_thread = threading.Thread(target=self.accept_connections, daemon=True)
            server_thread.start()
            
            self.log_message("Server started successfully!")
            self.log_message(f"Share this IP with patients: {self.get_local_ip()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {str(e)}")
            
    def stop_server(self):
        self.running = False
        
        # Close all client connections
        for client_socket in self.clients.values():
            try:
                client_socket.close()
            except:
                pass
        self.clients.clear()
        self.client_names.clear()
        self.update_clients_list()
        
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
            
        self.status_label.config(text="Status: Stopped", fg='red')
        self.info_label.config(text="")
        
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.send_button.config(state='disabled')
        
        self.log_message("Server stopped.")
        
    def accept_connections(self):
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                client_thread = threading.Thread(target=self.handle_client, 
                                               args=(client_socket, client_address), 
                                               daemon=True)
                client_thread.start()
            except Exception as e:
                if self.running:
                    self.log_message(f"Error accepting connection: {str(e)}")
                    
    def handle_client(self, client_socket, client_address):
        client_id = f"{client_address[0]}:{client_address[1]}"
        
        try:
            # Receive initial client info
            data = client_socket.recv(1024).decode('utf-8')
            client_info = json.loads(data)
            client_name = client_info.get('name', f"Patient-{len(self.clients)+1}")
            
            self.clients[client_id] = client_socket
            self.client_names[client_id] = client_name
            
            self.root.after(0, lambda: self.update_clients_list())
            self.root.after(0, lambda: self.log_message(f"Patient '{client_name}' connected from {client_address[0]}"))
            
            # Send welcome message
            welcome_msg = {
                'type': 'system',
                'message': f'Welcome {client_name}! You are now connected to the doctor.',
                'timestamp': datetime.datetime.now().isoformat()
            }
            client_socket.send(json.dumps(welcome_msg).encode('utf-8'))
            
            # Handle incoming messages
            while self.running:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                    
                message_data = json.loads(data)
                self.root.after(0, lambda msg=message_data: self.handle_incoming_message(msg, client_name))
                
        except Exception as e:
            error_msg = f"Error handling client {client_id}: {str(e)}"
            self.root.after(0, lambda msg=error_msg: self.log_message(msg))
        finally:
            # Clean up client connection
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.client_names:
                client_name = self.client_names[client_id]
                del self.client_names[client_id]
                self.root.after(0, lambda: self.log_message(f"Patient '{client_name}' disconnected"))
                self.root.after(0, lambda: self.update_clients_list())
            client_socket.close()
            
    def handle_incoming_message(self, message_data, client_name):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        message = message_data.get('message', '')
        
        self.messages_text.insert(tk.END, f"[{timestamp}] {client_name}: {message}\n")
        self.messages_text.see(tk.END)
        
        # Speak the patient's message using text-to-speech
        if self.tts_enabled and message and TTS_AVAILABLE:
            self.speak_message(message, client_name)
        
    def send_message(self, event=None):
        message = self.message_entry.get().strip()
        if not message:
            return
            
        if not self.clients:
            messagebox.showwarning("Warning", "No patients connected!")
            return
            
        # Send message to all connected clients
        timestamp = datetime.datetime.now().isoformat()
        message_data = {
            'type': 'doctor_message',
            'message': message,
            'timestamp': timestamp,
            'sender': 'Doctor'
        }
        
        message_json = json.dumps(message_data)
        disconnected_clients = []
        
        for client_id, client_socket in self.clients.items():
            try:
                client_socket.send(message_json.encode('utf-8'))
            except:
                disconnected_clients.append(client_id)
                
        # Remove disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.client_names:
                del self.client_names[client_id]
        self.update_clients_list()
        
        # Display message in doctor's chat
        timestamp_display = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp_display}] Doctor: {message}\n")
        self.messages_text.see(tk.END)
        
        self.message_entry.delete(0, tk.END)
        
    def update_clients_list(self):
        self.clients_listbox.delete(0, tk.END)
        for client_id, name in self.client_names.items():
            self.clients_listbox.insert(tk.END, f"{name} ({client_id})")
            
    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] SYSTEM: {message}\n")
        self.messages_text.see(tk.END)
        
    def speak_message(self, message, speaker_name=None):
        """Speak the message using text-to-speech with queue system"""
        if not self.tts_enabled:
            print("[TTS] TTS is disabled")
            return
        if not message:
            print("[TTS] No message to speak")
            return
        if not TTS_AVAILABLE:
            print("[TTS] TTS not available")
            return
        
        try:
            # Just speak the message directly, without speaker name prefix
            text_to_speak = message.strip()
            
            if not text_to_speak:
                print("[TTS] Message is empty after stripping")
                return
            
            print(f"[TTS] Queuing message: {text_to_speak[:50]}...")  # Debug output
            
            # Add message to queue
            with self.tts_lock:
                self.tts_queue.append(text_to_speak)
                queue_size = len(self.tts_queue)
                print(f"[TTS] Queue size: {queue_size}")  # Debug output
            
            # Start processing thread if not already running
            if not self.tts_processing:
                print("[TTS] Starting TTS processor")  # Debug output
                self._start_tts_processor()
            else:
                print("[TTS] Processor already running")  # Debug output
            
        except Exception as e:
            print(f"Error queuing message for TTS: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_tts_processor(self):
        """Start the TTS processing thread"""
        # Use a lock to prevent multiple processors from starting
        with self.tts_lock:
            if self.tts_processing:
                return
            self.tts_processing = True
        
        def tts_processor():
            """Process TTS queue continuously"""
            while True:
                try:
                    # Get next message from queue
                    with self.tts_lock:
                        if not self.tts_queue:
                            self.tts_processing = False
                            break
                        text_to_speak = self.tts_queue.pop(0)
                    
                    # Create a new engine instance for each message to avoid blocking
                    try:
                        import time
                        print(f"[TTS] Speaking: {text_to_speak[:50]}...")  # Debug output
                        engine = pyttsx3.init()
                        # Apply voice settings
                        if 'rate' in self.tts_voice_settings:
                            engine.setProperty('rate', self.tts_voice_settings['rate'])
                        if 'volume' in self.tts_voice_settings:
                            engine.setProperty('volume', self.tts_voice_settings['volume'])
                        if 'voice_id' in self.tts_voice_settings:
                            engine.setProperty('voice', self.tts_voice_settings['voice_id'])
                        
                        # Speak the message directly
                        engine.say(text_to_speak)
                        engine.runAndWait()
                        print(f"[TTS] Finished speaking")  # Debug output
                        engine.stop()  # Clean up
                        
                    except Exception as e:
                        print(f"TTS error speaking message: {e}")
                        import traceback
                        traceback.print_exc()
                    
                except Exception as e:
                    print(f"TTS processor error: {e}")
                    with self.tts_lock:
                        self.tts_processing = False
                    break
            
            with self.tts_lock:
                self.tts_processing = False
        
        # Start the processor thread
        tts_thread = threading.Thread(target=tts_processor, daemon=True)
        tts_thread.start()
    
    def toggle_tts(self):
        """Toggle text-to-speech on/off"""
        self.tts_enabled = not self.tts_enabled
        if self.tts_enabled:
            self.tts_toggle_button.config(text="🔊 TTS: ON", bg='#27ae60')
            # Clear queue when enabling
            with self.tts_lock:
                self.tts_queue.clear()
        else:
            self.tts_toggle_button.config(text="🔇 TTS: OFF", bg='#95a5a6')
            # Clear queue when disabling
            with self.tts_lock:
                self.tts_queue.clear()
    
    def get_local_ip(self):
        """Get the local IP address"""
        try:
            # Create a temporary socket to get local IP
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_socket.connect(("8.8.8.8", 80))
            local_ip = temp_socket.getsockname()[0]
            temp_socket.close()
            return local_ip
        except:
            return "127.0.0.1"
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DoctorServer()
    app.run()
