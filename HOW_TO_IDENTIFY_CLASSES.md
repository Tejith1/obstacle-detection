# 🔍 How to Identify Which Class Detects Which Obstacle

Your model has **22 classes** (class0 to class21), but they have generic names. Here's how to find out what each class actually detects:

---

## 🎯 **Method 1: Check Training Configuration (EASIEST)**

### Step 1: Find Your Training Dataset Configuration

Look for a `data.yaml` or `dataset.yaml` file that was used when training `best.pt`. It should look like this:

```yaml
# Example data.yaml
nc: 22  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', ...]  # class names
```

### Common Locations:
- Same folder where you trained the model
- In your dataset folder (e.g., `datasets/my_dataset/data.yaml`)
- Check your training command/notebook for the `--data` parameter

### If you find it:
The `names:` list maps directly to class IDs:
- names[0] = class0
- names[9] = class9
- etc.

---

## 🎯 **Method 2: Test with Real Objects (MOST PRACTICAL)**

Run the webcam detection and point your camera at known objects!

### Step-by-Step:

1. **Run the detector:**
   ```cmd
   python webcam_detect.py
   ```

2. **Point camera at different objects and note what class appears:**
   - Hold up your **phone** → Note which class it detects
   - Show a **person** (yourself) → Note the class
   - Point at a **car** through window → Note the class
   - Try a **bottle**, **book**, **chair**, etc.

3. **Create a mapping list:**
   ```
   class0 = person (tested with myself)
   class1 = bicycle (tested with my bike)
   class9 = traffic light (tested through window)
   ...
   ```

### I'll create a helper script for this! ↓

---

## 🎯 **Method 3: Check Training Logs/History**

Look for these files in your training directory:

### A. Training logs
```
runs/train/exp/
├── results.csv
├── hyp.yaml
└── opt.yaml  ← Check this file!
```

The `opt.yaml` or training logs might contain the dataset path which leads you to the data.yaml file.

### B. Weights metadata
Some training information might be saved in the model file itself. Try:
```python
import torch
checkpoint = torch.load('best.pt', map_location='cpu', weights_only=False)
print(checkpoint.keys())  # See what's stored
```

---

## 🎯 **Method 4: Check Dataset Folder Structure**

If you have the training dataset, look at the folder structure:

```
my_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml  ← This file has class names!
```

---

## 🎯 **Method 5: Ask the Person Who Trained It**

If someone else trained this model:
- Ask them for the `data.yaml` file
- Ask them what objects the model was trained to detect
- Ask them for the training command/script they used

---

## 💡 **Quick Solution: Use the Interactive Tester**

I've created `test_and_identify_classes.py` (see below) that helps you:
1. Run detection on webcam
2. Press keys to label what you're showing
3. Builds a class mapping automatically!

Run it with:
```cmd
python test_and_identify_classes.py
```

---

## 🎓 **Most Likely Classes (If Trained on Common Datasets)**

Based on 22 classes, your model might be trained on:

### Option A: Subset of COCO (Most Common)
Common obstacle detection classes from COCO:
- person, bicycle, car, motorcycle, bus, truck
- traffic light, fire hydrant, stop sign
- bench, bird, cat, dog, horse
- backpack, umbrella, handbag, suitcase
- bottle, chair, couch, potted plant
- etc.

### Option B: Custom Obstacle Dataset
Specialized for navigation/robotics:
- person, bicycle, car, motorcycle, bus, truck
- traffic light, stop sign, traffic cone, barrier
- pedestrian, cyclist, animal, obstacle
- pothole, speed bump, etc.

### Option C: VisDrone Dataset (Aerial View)
If for drones (22 classes in VisDrone):
- pedestrian, people, bicycle, car, van, truck
- tricycle, awning-tricycle, bus, motor
- others

---

## ✅ **Recommended Steps (In Order)**

1. **First**: Run the interactive tester (next section)
2. **Second**: Look for `data.yaml` in your training folders
3. **Third**: Check training logs for dataset path
4. **Fourth**: Test manually and document findings

---

## 🚀 **Next: Use the Interactive Class Identifier**

See the next script: `test_and_identify_classes.py`
