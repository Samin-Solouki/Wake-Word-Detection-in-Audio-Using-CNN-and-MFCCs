# Wake-Word-Detection-in-Audio-Using-CNN-and-MFCCs
Custom Wake Word Detection in Audio Using CNN and MFCCs / for the word "Samin"

### **Project Level**:
 Intermediate to Advanced – involves audio preprocessing, MFCC feature extraction, and deep learning model development for custom wake word detection.

*You may find the report of the project under the name "Wake-Word-Detection-in-Audio-Using-CNN-and-MFCCsi.PDF"*

---

##  Project Highlights

- **Project Goal**: Build a deep learning model capable of detecting a custom wake word in audio recordings.
- **Wake Word**: "Samin"
- **Techniques Used**: Audio preprocessing, feature extraction (MFCC), data augmentation, deep learning with CNN and Dense architectures.
- **Tools & Libraries**: Python, NumPy, Pandas, Librosa, TensorFlow/Keras, Matplotlib, Scikit-learn.

---

##  Objective

The aim of this project is to design and train a model that can accurately detect the presence of a predefined wake word in short audio clips. The model should distinguish between audio that contains the wake word and audio that does not.

---

##  Data Collection

- A dataset of audio samples was created, containing both positive and negative samples:
  - **Positive samples**: Audio recordings containing the wake word "Samin".
  - **Negative samples**: Audio clips of other unrelated words or background noise.
- All audio files were recorded in WAV format with a consistent sampling rate.
- Each audio clip was trimmed or padded to a fixed duration of **1 second**.

---

##  Preprocessing & Feature Extraction

- **Audio Processing**:
  - Audio normalized and converted to mono.
  - Clipping/padding to ensure uniform duration across samples.
- **Feature Extraction**:
  - Extracted **MFCCs (Mel Frequency Cepstral Coefficients)** as input features.
  - Used `librosa` for feature extraction.
  - Final input shape: (N, 13, T, 1), where N is the number of samples, 13 is the number of MFCC coefficients, T is the number of frames.

---

##  Data Augmentation

To improve generalization and reduce overfitting, data augmentation techniques were applied to the audio data:

- Adding background noise.
- Shifting the audio in time.
- Changing pitch or speed slightly.

This increased the size and diversity of the dataset.

---

##  Model Architecture

Two models were developed and tested:

### 1. CNN-Based Model
- Input layer for MFCCs.
- Multiple convolutional and pooling layers to capture local temporal patterns.
- Dense layers followed by a sigmoid output for binary classification.

### 2. Dense-Based Model
- Flattened MFCC features used as input.
- Several dense (fully connected) layers.
- Sigmoid output layer for binary prediction.

---

##  Training & Evaluation

- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam.
- **Metrics**: Accuracy, Precision, Recall, F1 Score.
- **Validation Split**: 20% of the dataset was used for validation.
- **Early Stopping** was implemented to avoid overfitting.

---

##  Results

- The CNN model outperformed the Dense-based model in both accuracy and robustness to noise.
- Final CNN model achieved:
  - Accuracy: ~94%
  - F1 Score: ~0.93
- Confusion matrix and ROC curve were plotted to visualize model performance.

---

## 💡 Conclusion

This project demonstrates the effectiveness of deep learning, particularly convolutional neural networks, in wake word detection tasks. The model successfully identifies the presence of the custom wake word "ثمین" in short audio clips, even in the presence of background noise. With further improvements and larger datasets, this approach can be scaled for real-time applications in voice-controlled systems.

