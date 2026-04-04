## What this code does

- Trains a binary classifier:
  - `0 = normal shopping / normal activity`
  - `1 = theft-like / suspicious activity`
- Works with **video files** organized into folders.
- Uses a **pretrained ResNet-18** to encode sampled frames from each clip.
- Aggregates frame features over time with a **GRU**.
- Can append simple handcrafted motion/context features:
  - mean optical-flow magnitude
  - max optical-flow magnitude
  - motion burst score
  - trajectory spread proxy
- Saves:
  - best model weights
  - confusion matrix
  - training curves
  - metrics JSON

## Expected folder structure
### Simplest structure

```text
data/
  train/
    normal/
      clip1.mp4
      clip2.mp4
    theft/
      clip3.mp4
      clip4.mp4
  val/
    normal/
    theft/
  test/
    normal/
    theft/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

### Standard binary training

```bash
python train_shopping_vs_theft.py \
  --data_root /path/to/data \
  --epochs 15 \
  --batch_size 4 \
  --num_frames 16 \
  --image_size 224 \
  --lr 1e-4
```
## Outputs

Saved under `runs/<timestamp>/`:

- `best_model.pt`
- `metrics.json`
- `confusion_matrix.png`
- `training_curves.png`
- `classification_report.txt`
