
# 🐱🐶 Cats vs Dogs Classification using VGG16
# remember- don't forget to enable gpu , use kaggle or google colab notebook
## 📌 Project Overview

This project implements an image classification model to distinguish between **cats and dogs** using **Transfer Learning** with VGG16.

The model uses a pre-trained convolutional neural network and is fine-tuned for binary image classification.

### 🔹 Model Performance

* **Training Accuracy:** 92%
* **Validation Accuracy:** 92%
* **Input Shape:** (224, 224, 3)
* **Label Mode:** Binary

---

## 🧠 Model Architecture – VGG16

![Image](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/network.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2ANNifzsJ7tD2kAfBXt3AzEg.png)

![Image](https://www.researchgate.net/publication/339851513/figure/fig4/AS%3A11431281194097782%401695955301032/Block-diagram-of-VGG-16-network.png)

![Image](https://www.researchgate.net/publication/360328250/figure/fig4/AS%3A1170322158026761%401656037954118/Illustration-of-VGG-16-high-level-block-diagram.png)

This project uses **VGG16**, a deep Convolutional Neural Network introduced by the Visual Geometry Group (Oxford).

### Why VGG16?

* Pre-trained on ImageNet
* Strong feature extraction capability
* Works well for transfer learning
* Suitable for small-to-medium datasets

---

## 📊 Dataset Structure

The dataset is organized using `flow_from_directory()`:

```
dataset/
│
├── train/
│   ├── cats/
│   └── dogs/
│
└── validation/
    ├── cats/
    └── dogs/
```

* Class Mode: **binary**
* Target Size: **(224, 224)**
* Color Mode: RGB

---

## 🔄 Data Preprocessing

Images were preprocessed using **ImageDataGenerator**:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

### 🔹 Preprocessing Steps

* Rescaling pixel values (0–255 → 0–1)
* Data augmentation (rotation, zoom, flip)
* Batch loading using `flow_from_directory()`

---

## 🏗️ Model Implementation

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 🔹 Loss Function & Optimizer

* Loss: Binary Crossentropy
* Optimizer: Adam
* Metrics: Accuracy

---

## 📈 Model Performance

| Metric              | Score |
| ------------------- | ----- |
| Training Accuracy   | 92%   |
| Validation Accuracy | 92%   |

The close training and validation accuracy indicates good generalization with minimal overfitting.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train.py
```

---

## 📂 Project Structure

```
├── dataset/
├── train.py
├── model/
├── requirements.txt
└── README.md
```

---

## 🎯 Applications

* Pet image recognition
* Basic computer vision projects
* Transfer learning demonstration
* Deep learning academic projects

---

## 🔮 Future Improvements

* Fine-tune deeper VGG16 layers
* Try EfficientNet / ResNet
* Add Grad-CAM visualization
* Deploy using Streamlit
* Convert to TensorFlow Lite for mobile apps

---

## 👨‍💻 Author

Affan Ahmad
Machine Learning Enthusiast

---

## ⭐ If you found this project helpful, consider giving it a star! 👐✌️

---

