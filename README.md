# Retinal Disease Classification using CNN
# [Train Model file](https://drive.google.com/drive/folders/1Vhay1rK-FAHX9-6f3j3-6FdqFunVj5pn)
## Project Overview

This project implements a Convolutional Neural Network (CNN) to automatically classify retinal fundus images into 8 eye disease categories. The model uses ConvNeXt-Base architecture with transfer learning and achieves a **Macro F1-Score of 0.7355** on the validation set.

### Disease Classes

| Code | Disease                                |
| ---- | -------------------------------------- |
| N    | Normal                                 |
| D    | Diabetic Retinopathy                   |
| G    | Glaucoma                               |
| C    | Cataract                               |
| A    | Age-related Macular Degeneration (AMD) |
| H    | Hypertensive Retinopathy               |
| M    | Myopia                                 |
| O    | Other diseases/abnormalities           |

## Project Structure

```
topic3/
├── 62FIT4ATI_Group 2_Topic 3.ipynb               # Main Jupyter notebook with complete workflow
├── README.md                                     # This file
├── report.docx                                   # Written report analyzing results
├── 62FIT4ATI_Group 2_Topic 3.pptx                # Slide content for presentation

```

## Requirements

### Hardware

- GPU with CUDA support (recommended)
- Minimum 8GB RAM
- ~10GB disk space for dataset

### Software Dependencies

```
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.0.0
tqdm>=4.62.0
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd topic3
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn Pillow tqdm jupyter
```

### 4. Download Dataset

Download the dataset from: [Google Drive Link](https://drive.google.com/drive/folders/1f5RCn1lxto-oAj6joSGyp4_R6wG4M6AP?usp=share_link)

Extract and organize the data:

```
data/
├── images/           # Contains all retinal fundus images
└── label_images.csv  # Labels for each image
```

## Reproduction Instructions

### Training the Model

1. Open `code.ipynb` in Jupyter Notebook or JupyterLab
2. Update the data path in Section 2 if needed
3. Run all cells sequentially

**Configuration options** (in Section 1):

```python
CONFIG = {
    "seed": 2024,
    "image_size": 256,
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 3e-4,
    "num_classes": 8,
    "patience": 5,          # Early stopping patience
    "model_name": "convnext_base"
}
```

### Expected Training Output

- Training time: ~30-45 minutes on GPU (NVIDIA RTX 3060 or similar)
- Best model saved to `best_model.pth`
- Expected Macro F1-Score: ~0.73-0.75

## Inference on New Images

### Using the trained model

```python
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Load model
def load_model(model_path, num_classes=8):
    model = models.convnext_base(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
def predict(image_path, model, device='cuda'):
    class_map = {0: 'N', 1: 'D', 2: 'G', 3: 'C', 4: 'A', 5: 'H', 6: 'M', 7: 'O'}

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return class_map[pred_idx], confidence * 100

# Usage
model = load_model('best_model.pth').to('cuda')
label, conf = predict('path/to/retinal_image.jpg', model)
print(f"Predicted: {label} ({conf:.2f}%)")
```

## Model Performance

| Metric               | Value  |
| -------------------- | ------ |
| Accuracy             | 73%    |
| Macro F1-Score       | 0.7355 |
| Best Validation Loss | 1.1451 |

### Per-class Performance

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| N (Normal)       | 0.73      | 0.75   | 0.74     |
| D (Diabetic)     | 0.63      | 0.62   | 0.62     |
| G (Glaucoma)     | 0.87      | 0.76   | 0.81     |
| C (Cataract)     | 0.76      | 0.91   | 0.83     |
| A (AMD)          | 0.70      | 0.70   | 0.70     |
| H (Hypertensive) | 0.88      | 0.73   | 0.80     |
| M (Myopia)       | 0.93      | 0.91   | 0.92     |
| O (Other)        | 0.47      | 0.47   | 0.47     |

## Key Optimization Techniques

1. **Data Augmentation**: Random flips, rotations, color jitter, affine transforms
2. **Class Weights**: Balanced weights to handle class imbalance
3. **Label Smoothing**: 0.1 smoothing factor for better generalization
4. **Test-Time Augmentation (TTA)**: Averaging predictions from original and flipped images
5. **Learning Rate Scheduling**: ReduceLROnPlateau with patience=2
6. **Early Stopping**: Patience=5 to prevent overfitting

## Authors

- Group Members: Trần Quang Huy - Nguyễn Thùy Anh - Mộc Khánh Duy
- Course: 62FIT4ATI
- Lecturer: Nguyễn Xuân Thắng
## License

This project is for educational purposes only.





