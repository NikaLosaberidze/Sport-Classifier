# Sport-Classifier
# 🏀⚽ Sports Video Classification - Basketball vs Football

This project classifies sports videos (Basketball and Football) using basic computer vision and machine learning techniques. It includes frame extraction, edge detection with Sobel filters, and KMeans clustering based on edge features.

---

## 📁 File Breakdown

### `extract_frames.py`

**Purpose:**  
Extracts 1 frame per second from videos stored in the `videos/` folder, organized by class (`Basketball/`, `Football/`). Saves frames to the `frames/` directory under the corresponding class.

**Key Features:**
- Automatically creates folders if needed
- Supports `.mp4`, `.avi`, `.mov` files

**How to use:**
```bash
python extract_frames.py


MAKE SURE YOUR FOLDER STRUCTURE LOOKS LIKE THIS:

videos/
├── Basketball/
│   ├── video1.mp4
│   └── ...
└── Football/
    ├── video1.mp4
    └── ...

