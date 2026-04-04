import torch
import cv2
import numpy as np
from pathlib import Path

from train_shopping_vs_theft import ResNetGRUClassifier  # adjust if name differs

MODEL_PATH = "/Users/alishbafarhan/Desktop./CS4452/shopping_vs_theft_project/runs/run_20260331_230307/best_model.pt"  # change if your saved model has different name
VIDEO_PATH = "/Users/alishbafarhan/Desktop./CS4452/shopping_vs_theft_project/data/test/Shoplifting/Shoplifting013_x264.mp4"  # change this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = ResNetGRUClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def load_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))  # to (T, C, H, W)
    return torch.tensor(frames, dtype=torch.float32)

# run prediction
frames = load_frames(VIDEO_PATH).unsqueeze(0).to(device)

with torch.no_grad():
    handcrafted = torch.zeros((1, 4), dtype=torch.float32).to(device)
    output = model(frames, handcrafted)
    prob = torch.sigmoid(output).item()

print("Probability of theft:", prob)

if prob > 0.5:
    print("Prediction: THEFT")
else:
    print("Prediction: NORMAL")