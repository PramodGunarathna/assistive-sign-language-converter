"""
Robust MediaPipe processor that handles timestamp issues
"""
import cv2
import numpy as np
import mediapipe as mp
import time
import threading


class RobustMediaPipeProcessor:
    """Robust MediaPipe processor that handles timestamp issues"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.holistic = None
        self.lock = threading.Lock()
        self.error_count = 0
        self.max_errors = 5  # Reset processor after 5 consecutive errors
        self.last_reset_time = time.time()
        self._initialize_holistic()
    
    def _initialize_holistic(self):
        """Initialize MediaPipe holistic processor"""
        try:
            if self.holistic:
                self.holistic.close()
                time.sleep(0.1)  # Small delay to ensure cleanup
            
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.error_count = 0  # Reset error count on successful initialization
            print("MediaPipe processor initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            self.holistic = None
    
    def detect_safe(self, image):
        """Safe detection that handles timestamp issues"""
        with self.lock:
            if self.holistic is None:
                # Try to reinitialize if processor is None
                self._initialize_holistic()
                if self.holistic is None:
                    return image, None
            
            try:
                # Convert image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                
                # Process with timeout protection
                results = self.holistic.process(rgb_image)
                
                # Convert back
                rgb_image.flags.writeable = True
                output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                # Reset error count on successful processing
                self.error_count = 0
                
                return output_image, results
                
            except Exception as e:
                self.error_count += 1
                print(f"MediaPipe detection error ({self.error_count}/{self.max_errors}): {e}")
                
                # Check if we need to reset the processor
                current_time = time.time()
                if (self.error_count >= self.max_errors or 
                    (current_time - self.last_reset_time) > 30):  # Reset every 30 seconds
                    print("Resetting MediaPipe processor due to repeated errors...")
                    self._force_reset()
                
                # Return original image and None results
                return image, None
    
    def draw_landmarks_safe(self, image, results):
        """Safely draw landmarks - only show hand landmarks, hide face and body"""
        if results is None:
            return image
            
        try:
            # Skip face landmarks - don't draw them
            # Skip pose landmarks - don't draw them
            
            # Only draw left hand connections
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )
            
            # Only draw right hand connections
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
        
        return image
    
    def extract_keypoints_safe(self, results):
        """Safely extract keypoints"""
        if results is None:
            # Return zeros if no results
            return np.zeros(33*4 + 468*3 + 21*3 + 21*3)
        
        try:
            # Extract pose landmarks
            if results.pose_landmarks:
                pose = np.array([[res.x, res.y, res.z, res.visibility] 
                               for res in results.pose_landmarks.landmark]).flatten()
            else:
                pose = np.zeros(33*4)
            
            # Extract face landmarks
            if results.face_landmarks:
                face = np.array([[res.x, res.y, res.z] 
                               for res in results.face_landmarks.landmark]).flatten()
            else:
                face = np.zeros(468*3)
            
            # Extract left hand landmarks
            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] 
                             for res in results.left_hand_landmarks.landmark]).flatten()
            else:
                lh = np.zeros(21*3)
            
            # Extract right hand landmarks
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] 
                             for res in results.right_hand_landmarks.landmark]).flatten()
            else:
                rh = np.zeros(21*3)
            
            return np.concatenate([pose, face, lh, rh])
            
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return np.zeros(33*4 + 468*3 + 21*3 + 21*3)
    
    def _force_reset(self):
        """Force reset the MediaPipe processor"""
        try:
            if self.holistic:
                self.holistic.close()
                time.sleep(0.2)  # Longer delay for complete cleanup
            self.holistic = None
            self.error_count = 0
            self.last_reset_time = time.time()
            self._initialize_holistic()
            print("MediaPipe processor force reset completed")
        except Exception as e:
            print(f"Error during force reset: {e}")
    
    def reset_processor(self):
        """Manually reset the processor (call this between recording sessions)"""
        print("Manually resetting MediaPipe processor...")
        self._force_reset()
    
    def close(self):
        """Close the processor"""
        try:
            if self.holistic:
                self.holistic.close()
                time.sleep(0.1)
            self.holistic = None
            print("MediaPipe processor closed successfully")
        except Exception as e:
            print(f"Error closing MediaPipe processor: {e}")


def prob_viz_safe(res, actions, input_frame, colors):
    """Safely visualize prediction probabilities - only show predicted word at top"""
    try:
        output_frame = input_frame.copy()
        
        # Find the predicted action (highest probability)
        if len(res) > 0 and len(actions) > 0:
            predicted_index = np.argmax(res)
            predicted_action = actions[predicted_index]
            confidence = res[predicted_index]
            
            # Only show the predicted word at the top of the screen
            # Create a background rectangle for the predicted word
            cv2.rectangle(output_frame, (0, 0), (640, 60), (0, 0, 0), -1)  # Black background
            cv2.rectangle(output_frame, (0, 0), (640, 60), (0, 255, 0), 3)  # Green border
            
            # Display the predicted word prominently
            text = f"PREDICTED: {predicted_action.upper()}"
            confidence_text = f"Confidence: {confidence:.1%}"
            
            # Main prediction text (large and bold)
            cv2.putText(output_frame, text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Confidence text (smaller)
            cv2.putText(output_frame, confidence_text, (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return output_frame
    except Exception as e:
        print(f"Error in prob_viz: {e}")
        return input_frame

