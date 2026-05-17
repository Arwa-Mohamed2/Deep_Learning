# 🚦 Traffic Sign Classification — CNN

**Course:** Deep Learning  
**Dataset:** GTSRB — German Traffic Sign Recognition Benchmark  
**Framework:** TensorFlow / Keras · Google Colab  
**Classes:** 43 traffic sign types · ~50,000 images

---

## Problem Description

Traffic sign recognition is a core component of autonomous driving and driver-assistance systems. This project builds and trains a **Convolutional Neural Network (CNN)** to classify 43 types of German traffic signs from the GTSRB benchmark dataset.

Two experiments are conducted to compare the effect of different optimisers on model performance — all other hyperparameters are held constant.

---

## Dataset

- **Name:** GTSRB — German Traffic Sign Recognition Benchmark
- **Source:** [Kaggle — meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Size:** ~50,000 images across 43 classes
- **Format:** RGB images of varying sizes, organised into class subfolders (0–42)
- **Split:** 80% training / 20% testing (stratified)

---

## Model Architecture

```
Input (64×64×3)
  └─ Conv Block 1 : Conv2D(32)  → BatchNorm → MaxPool  →  (32×32×32)
  └─ Conv Block 2 : Conv2D(64)  → BatchNorm → MaxPool  →  (16×16×64)
  └─ Conv Block 3 : Conv2D(128) → BatchNorm → MaxPool  →  (8×8×128)
  └─ Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(43, Softmax)
```

**Enhancements applied:**
-  **Batch Normalisation** — stabilises and speeds up training between layers
-  **Dropout (0.5)** — prevents overfitting in the fully connected head
-  **Data Augmentation** — rotation, zoom, shift, shear applied to training set only via `ImageDataGenerator`

---

## Experiments

| | Model A | Model B |
|---|---|---|
| **Optimiser** | Adam | SGD |
| **Learning Rate** | 0.001 | 0.01 |
| **Momentum** | — | 0.9 |
| **Epochs** | 8 | 8 |
| **Batch Size** | 64 | 64 |

> Only the optimiser changes between experiments. Architecture, epochs, and batch size are identical — this isolates the effect of the optimiser on performance.

---

## Results

| Model | Accuracy | Loss |
|---|---|---|
| Model A — Adam (lr=0.001) | 0.9923% | 0.0209 |
| Model B — SGD (lr=0.01, momentum=0.9) | 0.9935% | 0.0185 |

Best model: <!-- Model B (SGD) -->
Difference: <!-- Model B outperformed Model B by .11% -->

---

## 📈 Visualisations

All plots are saved automatically to Google Drive under `traffic_sign_project/results/`:

| File | Description |
|---|---|
| `sample_images.png` | Sample traffic signs from the dataset |
| `class_distribution.png` | Bar chart of image count per class |
| `training_curves.png` | Loss & accuracy curves for both models (2×2 grid) |
| `confusion_matrix.png` | 43×43 confusion matrix for the best model |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/iarwamohamed/traffic-sign-classification.git
cd traffic-sign-classification
```

### 2. Open in Google Colab

Notebook Link: https://colab.research.google.com/drive/10dZRmkv-iQBIE7BYRPjUe9GZzPVxRz62?usp=sharing

### 3. Set up Kaggle credentials
In **Cell 4**, replace the placeholders with your own Kaggle credentials:
```python
KAGGLE_TOKEN    = "YOUR_KAGGLE_TOKEN"
KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
```
Get your token at: kaggle.com → Account → API → **Create New Token**

### 4. Run all cells in order

| Cell | Purpose |
|---|---|
| Cell 2 | Mount Google Drive |
| Cell 4 | Configure Kaggle API credentials |
| Cell 5 | Download & unzip GTSRB dataset (~300 MB) |
| Cell 7 | Imports & libraries |
| Cell 9 | Load images from disk into numpy arrays |
| Cell 10 | Normalise · one-hot encode · train/test split |
| Cell 12 | Define class names (all 43 signs) |
| Cell 13 | Class distribution chart |
| Cell 14 | Sample images display |
| Cell 15 | Data augmentation setup |
| Cell 17 | Build CNN architecture |
| Cell 19 | **Experiment A** — Train with Adam |
| Cell 21 | **Experiment B** — Train with SGD |
| Cell 23 | Training curves (loss & accuracy, 2×2 grid) |
| Cell 25 | Results comparison table |
| Cell 27 | Classification report — best model |
| Cell 29 | Confusion matrix — best model |
| Cell 31 | Save both models to Google Drive |
| Cell 33 | Test on your own uploaded image |

### 5. Test on a custom image (Cell 33)
1. Download any traffic sign image from the web
2. In Colab, open the **Files panel** (📁 icon on the left sidebar) → click **Upload**
3. Right-click the uploaded file → **Copy path**
4. Paste it into Cell 33:
```python
IMAGE_PATH = '/content/your_image.png'
```
5. Run the cell — the model displays its **top-3 predictions** with confidence scores as a bar chart

---

## Repository Structure

```
traffic-sign-classification/
│
├── DL_Proj_Traffic.ipynb        # Main notebook
├── README.md                    # This file
└── results/                     # Auto-generated after training (saved to Drive)
    ├── sample_images.png
    ├── class_distribution.png
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## 👩‍💻 Author

**Arwa Mohamed**  
Deep Learning Course
