"""
Model utility functions for sign language recognition
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Attention, Add, Input,
    GlobalAveragePooling1D, Concatenate, Multiply
)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


class SignLanguageModel:
    """Sign Language Recognition Model"""
    
    def __init__(self, actions, sequence_length=30, model_type='improved'):
        self.actions = actions
        self.sequence_length = sequence_length
        self.model = None
        self.label_map = {label: num for num, label in enumerate(actions)}
        self.model_type = model_type  # 'basic', 'improved', or 'advanced'
    
    def create_basic_model(self, input_shape):
        """Create basic LSTM model architecture (original)"""
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=False, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.actions), activation='softmax'))
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
    
    def create_model(self, input_shape):
        """Create model based on specified architecture type"""
        if self.model_type == 'basic':
            return self.create_basic_model(input_shape)
        elif self.model_type == 'advanced':
            return self.create_advanced_model(input_shape)
        else:  # 'improved' (default)
            return self.create_improved_model(input_shape)
    
    def create_improved_model(self, input_shape):
        """Create improved LSTM model architecture with bidirectional layers, attention, and regularization"""
        # Input layer
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # First bidirectional LSTM layer with dropout
        lstm1 = Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh', 
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            name='bidirectional_lstm_1'
        )(inputs)
        lstm1 = BatchNormalization(name='bn_1')(lstm1)
        lstm1 = Dropout(0.3, name='dropout_1')(lstm1)
        
        # Second bidirectional LSTM layer
        lstm2 = Bidirectional(
            LSTM(256, return_sequences=True, activation='tanh',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            name='bidirectional_lstm_2'
        )(lstm1)
        lstm2 = BatchNormalization(name='bn_2')(lstm2)
        lstm2 = Dropout(0.4, name='dropout_2')(lstm2)
        
        # Third bidirectional LSTM layer
        lstm3 = Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            name='bidirectional_lstm_3'
        )(lstm2)
        lstm3 = BatchNormalization(name='bn_3')(lstm3)
        lstm3 = Dropout(0.3, name='dropout_3')(lstm3)
        
        # Attention mechanism
        attention_weights = Dense(1, activation='softmax', name='attention_weights')(lstm3)
        attention_output = Multiply(name='attention_output')([lstm3, attention_weights])
        
        # Global average pooling to get fixed-size representation
        pooled = GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
        
        # Dense layers with residual connections
        dense1 = Dense(256, activation='relu', 
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                      name='dense_1')(pooled)
        dense1 = BatchNormalization(name='bn_dense_1')(dense1)
        dense1 = Dropout(0.5, name='dropout_dense_1')(dense1)
        
        dense2 = Dense(128, activation='relu',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                      name='dense_2')(dense1)
        dense2 = BatchNormalization(name='bn_dense_2')(dense2)
        dense2 = Dropout(0.4, name='dropout_dense_2')(dense2)
        
        # Final classification layer
        outputs = Dense(len(self.actions), activation='softmax', 
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                       name='output')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='improved_sign_language_model')
        
        # Compile with improved optimizer and learning rate
        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        return model
    
    def create_advanced_model(self, input_shape):
        """Create an even more advanced LSTM model with residual connections and multi-head attention"""
        # Input layer
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # First bidirectional LSTM with residual connection
        lstm1 = Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            name='bidirectional_lstm_1'
        )(inputs)
        lstm1 = BatchNormalization(name='bn_1')(lstm1)
        lstm1 = Dropout(0.2, name='dropout_1')(lstm1)
        
        # Second bidirectional LSTM
        lstm2 = Bidirectional(
            LSTM(256, return_sequences=True, activation='tanh',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            name='bidirectional_lstm_2'
        )(lstm1)
        lstm2 = BatchNormalization(name='bn_2')(lstm2)
        lstm2 = Dropout(0.3, name='dropout_2')(lstm2)
        
        # Residual connection
        residual = Dense(512, activation='linear', name='residual_projection')(lstm1)
        lstm2_residual = Add(name='residual_connection')([lstm2, residual])
        
        # Third bidirectional LSTM
        lstm3 = Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            name='bidirectional_lstm_3'
        )(lstm2_residual)
        lstm3 = BatchNormalization(name='bn_3')(lstm3)
        lstm3 = Dropout(0.3, name='dropout_3')(lstm3)
        
        # Multi-head attention mechanism
        attention_heads = []
        for i in range(3):  # 3 attention heads
            attention_weights = Dense(1, activation='softmax', 
                                   name=f'attention_weights_{i}')(lstm3)
            attention_output = Multiply(name=f'attention_output_{i}')([lstm3, attention_weights])
            attention_heads.append(attention_output)
        
        # Concatenate attention heads
        multi_head_attention = Concatenate(name='multi_head_attention')(attention_heads)
        
        # Global pooling
        pooled = GlobalAveragePooling1D(name='global_avg_pool')(multi_head_attention)
        
        # Dense layers with residual connections
        dense1 = Dense(512, activation='relu',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                      name='dense_1')(pooled)
        dense1 = BatchNormalization(name='bn_dense_1')(dense1)
        dense1 = Dropout(0.5, name='dropout_dense_1')(dense1)
        
        dense2 = Dense(256, activation='relu',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                      name='dense_2')(dense1)
        dense2 = BatchNormalization(name='bn_dense_2')(dense2)
        dense2 = Dropout(0.4, name='dropout_dense_2')(dense2)
        
        dense3 = Dense(128, activation='relu',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                      name='dense_3')(dense2)
        dense3 = BatchNormalization(name='bn_dense_3')(dense3)
        dense3 = Dropout(0.3, name='dropout_dense_3')(dense3)
        
        # Final classification layer
        outputs = Dense(len(self.actions), activation='softmax',
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                       name='output')(dense3)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='advanced_sign_language_model')
        
        # Compile with advanced optimizer
        optimizer = AdamW(learning_rate=0.0008, weight_decay=1e-4, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        return model
    
    def load_data(self, data_path):
        """Load training data from directory"""
        sequences, labels = [], []
        
        print(f"Loading data from: {data_path}")
        print(f"Actions to load: {self.actions}")
        
        for action in self.actions:
            action_path = os.path.join(data_path, action)
            print(f"Checking action path: {action_path}")
            
            if not os.path.exists(action_path):
                print(f"Action path does not exist: {action_path}")
                continue
            
            # Get all sequence directories
            sequence_dirs = [d for d in os.listdir(action_path) if d.isdigit()]
            sequence_dirs.sort(key=int)
            print(f"Found {len(sequence_dirs)} sequences for action '{action}'")
            
            for sequence in sequence_dirs:
                sequence_path = os.path.join(action_path, sequence)
                print(f"Processing sequence: {sequence_path}")
                
                # Check if sequence has enough frames
                frame_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
                frame_files.sort(key=lambda x: int(x.split('.')[0]))
                
                if len(frame_files) < self.sequence_length:
                    print(f"Warning: Sequence {sequence} has only {len(frame_files)} frames, skipping")
                    continue
                
                window = []
                for frame_num in range(self.sequence_length):
                    frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    if os.path.exists(frame_path):
                        res = np.load(frame_path)
                        window.append(res)
                    else:
                        print(f"Warning: Frame {frame_num} not found in sequence {sequence}")
                        # Use zeros if frame is missing
                        window.append(np.zeros(1662))  # Default keypoint size
                
                if len(window) == self.sequence_length:
                    sequences.append(window)
                    labels.append(self.label_map[action])
                    print(f"Added sequence {sequence} for action '{action}'")
                else:
                    print(f"Skipped sequence {sequence} - incomplete")
        
        print(f"Total sequences loaded: {len(sequences)}")
        print(f"Total labels: {len(labels)}")
        
        return np.array(sequences), np.array(labels)
    
    def prepare_data(self, X, y, test_size=0.05):
        """Prepare data for training"""
        y_categorical = to_categorical(y).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size)
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_test, y_test, epochs=None, log_dir='Logs'):
        """Train the improved model with advanced callbacks and learning rate scheduling"""
        # Set default epochs if none provided
        if epochs is None:
            epochs = 2000
            print(f"DEBUG: No epochs provided, using default: {epochs}")
        
        print(f"DEBUG: SignLanguageModel.train() called with epochs={epochs}")
        
        if self.model is None:
            self.model = self.create_model((self.sequence_length, X_train.shape[2]))
        
        # Enhanced callbacks for better training
        tb_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        # Early stopping with more conservative patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=max(50, epochs // 20),  # More conservative patience
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4
        )
        
        # Learning rate reduction on plateau
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-7,
            verbose=1,
            cooldown=10
        )
        
        print(f"DEBUG: Starting model.fit() with epochs={epochs}")
        print(f"DEBUG: Using {self.model_type} architecture")
        if self.model_type == 'improved':
            print("DEBUG: Features: Bidirectional LSTM, Attention mechanism, Batch Normalization, Dropout, L1/L2 regularization")
        elif self.model_type == 'advanced':
            print("DEBUG: Features: Bidirectional LSTM, Multi-head attention, Residual connections, Advanced regularization")
        else:
            print("DEBUG: Features: Basic LSTM architecture")
        
        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            validation_data=(X_test, y_test),
            callbacks=[tb_callback, early_stopping, lr_scheduler],
            verbose=1,
            batch_size=32  # Optimal batch size for LSTM training
        )
        
        print(f"DEBUG: Model training completed. Actual epochs: {len(history.history['loss'])}")
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        yhat = self.model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat_classes = np.argmax(yhat, axis=1).tolist()
        
        accuracy = accuracy_score(ytrue, yhat_classes)
        confusion_matrix = multilabel_confusion_matrix(ytrue, yhat_classes)
        
        return accuracy, confusion_matrix
    
    def predict(self, sequence):
        """Make prediction on a sequence"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(np.expand_dims(sequence, axis=0))[0]
    
    def save_model(self, filepath):
        """Save the trained model with metadata"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Save the model
        self.model.save(filepath)
        
        # Save metadata (actions and other info)
        metadata_path = filepath.replace('.h5', '_metadata.json')
        import json
        metadata = {
            'actions': self.actions,
            'sequence_length': self.sequence_length,
            'input_shape': (self.sequence_length, 1662),
            'num_classes': len(self.actions)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {filepath}")
        print(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        
        # Try to load metadata
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.actions = metadata.get('actions', [])
                self.sequence_length = metadata.get('sequence_length', 30)
                print(f"Loaded model metadata: {metadata}")
        else:
            print("No metadata found, trying to infer from model structure")
            # Try to infer from model output shape
            output_shape = self.model.output_shape[1]
            print(f"Model output shape: {output_shape}")
            
            # Create generic action names based on the number of classes
            if output_shape == 2:
                self.actions = ['Action_1', 'Action_2']
            elif output_shape == 3:
                self.actions = ['Action_1', 'Action_2', 'Action_3']
            elif output_shape == 6:
                self.actions = ['Action_1', 'Action_2', 'Action_3', 'Action_4', 'Action_5', 'Action_6']
            else:
                # Create generic names for any number of classes
                self.actions = [f'Action_{i+1}' for i in range(output_shape)]
            
            print(f"Using inferred actions: {self.actions}")
        
        return self.model
