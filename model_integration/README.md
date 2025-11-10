# Sign Language Recognition System - New Project

A complete sign language recognition system with real-time word prediction and sentence generation using advanced LSTM architecture and AI enhancement.

## 🚀 Features

### **Real-time Word Prediction**
- Live sign language recognition
- Hand visibility validation (only predicts when hands are visible)
- Clean interface showing only hand landmarks
- Improved LSTM architecture with bidirectional layers
- Attention mechanism for better accuracy

### **Sentence Prediction**
- Captures multiple sign words
- Generates meaningful 6+ word sentences using Gemini AI
- Automatic sentence generation when hands are removed
- Medical/health context enhancement
- Fallback system for offline use

### **Advanced Architecture**
- **Basic**: Simple LSTM layers (fastest)
- **Improved**: Bidirectional LSTM + Attention + Regularization (recommended)
- **Advanced**: Multi-head attention + Residual connections (most accurate)

## 📁 Project Structure

```
New Project01/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── gemini_integration.py           # Gemini AI integration
└── src/
    ├── real_time_prediction.py     # Real-time word prediction module
    ├── sentence_prediction.py      # Sentence prediction module
    ├── utils/
    │   ├── model_utils.py          # Model utilities and architectures
    │   └── robust_mediapipe.py     # MediaPipe processing
    └── models/
        └── trained/                # Directory for trained models
```

## 🛠️ Installation

1. **Clone or download this project**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your trained models in `src/models/trained/`**
   - Models should be `.h5` files
   - Include corresponding `_metadata.json` files

## 🎯 Usage

### **Running the Application**
```bash
python main.py
```

### **Main Interface**
- **🔮 Real-time Word Prediction**: Live sign language recognition
- **📝 Sentence Prediction**: Generate meaningful sentences from signs

### **Real-time Word Prediction**
1. Load a trained model
2. Click "Start Prediction"
3. Show your hands in front of the camera
4. Sign words - predictions appear in real-time
5. Only hand landmarks are shown (no face/body clutter)

### **Sentence Prediction**
1. Load a trained model
2. Click "Start Prediction"
3. Sign multiple words in front of the camera
4. Remove both hands from camera view
5. System automatically generates a 6+ word meaningful sentence

## 🧠 Model Architecture Options

### **Basic Architecture**
- Simple LSTM layers
- Fastest training and inference
- Good for quick prototyping

### **Improved Architecture** (Recommended)
- Bidirectional LSTM layers
- Attention mechanism
- Batch normalization and dropout
- L1/L2 regularization
- Better accuracy with balanced performance

### **Advanced Architecture**
- Multi-head attention mechanism
- Residual connections
- Enhanced regularization
- Highest accuracy but slower training

## 🔧 Configuration

### **Gemini AI Setup**
- Update the API key in `src/sentence_prediction.py`
- Default key: `AIzaSyBPM7O_AdBuof7DiCahobiIhFUmOEz8q9U`
- Replace with your own Gemini API key

### **Model Requirements**
- Input shape: (30, 1662) - 30 frames, 1662 keypoints
- Output: Softmax probabilities for each action
- Supported formats: Keras H5 models

## 📊 Key Improvements

### **Hand Visibility Validation**
- Only makes predictions when hands are visible
- Prevents false predictions from old data
- Clear visual feedback when hands not detected

### **6+ Word Sentence Generation**
- Always generates meaningful sentences with 6+ words
- Uses Gemini AI for intelligent enhancement
- Medical/health context for better communication
- Fallback system for offline use

### **Clean Interface**
- Only shows hand landmarks (no face/body)
- Focused prediction display
- Better user experience

## 🚨 Troubleshooting

### **Common Issues**

1. **Camera not opening**
   - Check camera permissions
   - Ensure no other applications are using the camera

2. **Model loading errors**
   - Verify model file exists and is valid
   - Check model format (should be .h5)

3. **Gemini API errors**
   - Verify API key is correct
   - Check internet connection
   - System will fallback to basic enhancement

4. **No predictions**
   - Ensure hands are visible in camera
   - Check lighting conditions
   - Verify model is properly trained

### **Debug Information**
- Console output shows detailed debug information
- Hand visibility status is logged
- Prediction confidence scores are displayed

## 📈 Performance Tips

1. **Good lighting** for better hand detection
2. **Clear background** to avoid distractions
3. **Consistent hand positioning** for better accuracy
4. **Use improved or advanced architecture** for better results

## 🔄 Updates and Maintenance

- Models can be retrained and replaced in `src/models/trained/`
- Architecture can be changed in `src/utils/model_utils.py`
- Gemini prompts can be modified in `gemini_integration.py`

## 📝 License

This project is part of the Sign Language Recognition System. Please ensure you have the necessary permissions to use the trained models and API keys.

## 🤝 Support

For issues and questions:
1. Check the console output for debug information
2. Verify all dependencies are installed
3. Ensure models are properly formatted
4. Check camera and API connectivity
