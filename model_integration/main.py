"""
Sign Language Recognition System - New Project
Main Application Entry Point
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.real_time_prediction import RealTimePredictor
from src.sentence_prediction import SentencePredictor


class SignLanguageApp:
    """Main application for Sign Language Recognition System"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition System - New Project")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Center the window
        self.center_window()
        
        # Create main interface
        self.create_main_interface()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_main_interface(self):
        """Create the main application interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Sign Language Recognition System", 
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Description
        desc_text = ("Welcome to the Sign Language Recognition System!\n\n"
                   "This application provides two main functionalities:\n"
                   "1. Real-time Word Prediction - Live sign language recognition\n"
                   "2. Sentence Prediction - Generate meaningful sentences from signs\n\n"
                   "Features:\n"
                   "• Improved LSTM architecture with bidirectional layers\n"
                   "• Attention mechanism for better accuracy\n"
                   "• Hand visibility validation\n"
                   "• 6+ word sentence generation with AI\n"
                   "• Clean interface with only hand landmarks\n\n"
                   "Select a module to get started:")
        
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.LEFT)
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Real-time Prediction button
        self.realtime_btn = ttk.Button(
            buttons_frame, 
            text="🔮 Real-time Word Prediction", 
            command=self.open_realtime_prediction,
            width=25
        )
        self.realtime_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Sentence Prediction button
        self.sentence_btn = ttk.Button(
            buttons_frame, 
            text="📝 Sentence Prediction", 
            command=self.open_sentence_prediction,
            width=25
        )
        self.sentence_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Exit button
        self.exit_btn = ttk.Button(
            buttons_frame, 
            text="❌ Exit", 
            command=self.exit_application,
            width=25
        )
        self.exit_btn.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        
        # Check system status
        self.check_system_status(status_frame)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
    
    def check_system_status(self, parent):
        """Check and display system status"""
        status_text = "System Status:\n"
        
        # Check if required directories exist
        required_dirs = ['src/models/trained', 'src/utils']
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                status_text += f"✓ {dir_path} - Ready\n"
            else:
                status_text += f"✗ {dir_path} - Missing\n"
        
        # Check for trained models
        if os.path.exists('src/models/trained'):
            model_count = len([f for f in os.listdir('src/models/trained') if f.endswith('.h5')])
            status_text += f"🧠 Trained models: {model_count}\n"
        
        status_label = ttk.Label(parent, text=status_text, justify=tk.LEFT)
        status_label.grid(row=0, column=0, sticky=tk.W)
    
    def open_realtime_prediction(self):
        """Open real-time prediction module"""
        try:
            predictor = RealTimePredictor()
            predictor.start_prediction_gui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open real-time prediction: {str(e)}")
    
    def open_sentence_prediction(self):
        """Open sentence prediction module"""
        try:
            predictor = SentencePredictor()
            predictor.start_sentence_prediction_gui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open sentence prediction: {str(e)}")
    
    def exit_application(self):
        """Exit the application"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.quit()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main function to run the application"""
    try:
        app = SignLanguageApp()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()