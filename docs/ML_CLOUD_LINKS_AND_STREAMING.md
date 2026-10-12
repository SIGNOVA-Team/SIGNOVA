# Machine Learning Cloud Resource Links & Access Guide
**Project:** SIGNOVA  
**Datasets:** How2Sign (Continuous ASL) & ArabSign (Continuous ArSL)  
**Goal:** Direct streaming & cloud execution without storing raw multi-terabyte videos locally.

---

## 1. Quick Access Summary

| Dataset | Modality | Size | Storage Type | Direct Link / ID | Access Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **How2Sign** | RGB Front Sentence Clips (Train) | ~135 GB | Google Drive | `1dYey1F_SeHets-UO8F9cE3VMhRBO-6e0` | Direct GDrive Stream / Colab |
| **How2Sign** | RGB Front Sentence Clips (Val) | ~7 GB | Google Drive | `1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1` | Direct GDrive Stream / Colab |
| **How2Sign** | RGB Front Sentence Clips (Test) | ~11 GB | Google Drive | `1d6GHqu0_8IGiKbu3sTZHtMb0DGhbHSMu` | Direct GDrive Stream / Colab |
| **How2Sign** | 2D OpenPose Keypoints (Train) | ~70 GB | Google Drive | `1lnsDN-LxcsroOmetdG5_sXYXZ7setlS4` | Direct GDrive Stream / Colab |
| **How2Sign** | 2D OpenPose Keypoints (Val) | ~3.8 GB | Google Drive | `1aOhRknNWj8APdxHmwJdQrMo5xuIGNXxM` | Direct GDrive Stream / Colab |
| **How2Sign** | 2D OpenPose Keypoints (Test) | ~5.8 GB | Google Drive | `1quj8Ipm56pH65KAKK3Pc-sqZ0ozw2gSe` | Direct GDrive Stream / Colab |
| **ArabSign** | Color RGB Videos (All 9,335) | ~60 GB | Google Drive | `1twRzL8fqjbq5dp6jV-V00-O8gD1BZMTN` | [Folder Link](https://drive.google.com/drive/folders/1twRzL8fqjbq5dp6jV-V00-O8gD1BZMTN) (Request access) |
| **ArabSign** | 3D Skeleton Joints (25 joints) | ~5 GB | Google Drive | `18QN4X2vnWZ5PvNVik6TJJ7N606pRMvwA` | [Folder Link](https://drive.google.com/drive/folders/18QN4X2vnWZ5PvNVik6TJJ7N606pRMvwA) (Request access) |
| **ArabSign** | Depth Maps (Kinect V2) | ~40 GB | Google Drive | `1WJC9Bo9CSP0tVmPJlkTb-KA_n0VWp7LZ` | [Folder Link](https://drive.google.com/drive/folders/1WJC9Bo9CSP0tVmPJlkTb-KA_n0VWp7LZ) (Request access) |
| **ArabSign** | Official GitHub & Splits Code | ~1.5 MB | GitHub | [Hamzah-Luqman/ArabSign](https://github.com/Hamzah-Luqman/ArabSign) | Direct public download |

---

## 2. Direct Cloud Streaming / Ingestion Code

You can stream or mount any of these directly in a **Google Colab Notebook** or **Kaggle Notebook** without downloading files to your computer.

### Step A: Stream in Google Colab (Zero Local Disk Space)

```python
# 1. Mount Google Drive in Google Colab
from google.colab import drive
drive.mount('/content/drive')

# 2. Or download a specific split directly into cloud memory using gdown
!pip install -q gdown

# Example: Download How2Sign Validation Clips (7 GB directly into Colab's ephemeral cloud SSD)
!gdown 1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1 -O /content/how2sign_val_clips.zip

# Example: Download ArabSign Code & Splits
!wget https://github.com/Hamzah-Luqman/ArabSign/raw/main/ArabSign-EncoderDec.zip
!unzip -q ArabSign-EncoderDec.zip -d /content/arabsign_code
```

### Step B: Direct Video Streaming Dataset (Python PyTorch / OpenCV)

To stream and decode video frames directly from remote URLs into GPU RAM without saving them to hard disk:

```python
import cv2
import torch
from torch.utils.data import IterableDataset

class RemoteVideoStreamer(IterableDataset):
    """Streams video frames over HTTP directly into memory."""
    def __init__(self, video_urls, target_frames=32, target_size=(224, 224)):
        self.video_urls = video_urls
        self.target_frames = target_frames
        self.target_size = target_size

    def __iter__(self):
        for url in self.video_urls:
            cap = cv2.VideoCapture(url)
            frames = []
            while cap.isOpened() and len(frames) < self.target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            if len(frames) > 0:
                # Convert to PyTorch Tensor [C, T, H, W]
                tensor = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
                yield tensor
```

---

## 3. ArabSign Academic Access Request (One-Time Setup)

The ArabSign raw Google Drive folders are protected by KFUPM academic policy. Send this 30-second email to gain instant access to the Google Drive links:

- **Recipient:** `hluqman@kfupm.edu.sa` (Dr. Hamzah Luqman, KFUPM)
- **Subject:** `Access Request for ArabSign Dataset - SIGNOVA Project`
- **Body:**
  > Dear Dr. Luqman,  
  > I am working on the SIGNOVA project studying Arabic Sign Language recognition and continuous translation benchmarks.  
  > Could you please grant access to the ArabSign Google Drive repository for:  
  > 1. 3D Skeleton Joint Points (`18QN4X2vnWZ5PvNVik6TJJ7N606pRMvwA`)  
  > 2. Color RGB Videos (`1twRzL8fqjbq5dp6jV-V00-O8gD1BZMTN`)  
  >  
  > Google Account Email: [YOUR_GMAIL_HERE]  
  > Best regards,  
  > [YOUR NAME]
