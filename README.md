## Example of using ResNet18 and RetinaFace Slim `tflite` models to detect faces with landmarks

### General

This project showcases the utilization of two packages, namely resnet18 and slim, for face detection with landmarks.
Each package contains a main.py file that demonstrates the functionality of the respective model.

This project contains two packages - `resnet18` and `slim` - for detecting faces and facial landmarks in images using
TensorFlow Lite models.

### Original models

ResNet18 model was taken from this repo: https://github.com/DefTruth/torchlm

Slim model was taken from this repo: https://github.com/biubug6/Face-Detector-1MB-with-landmark

### Convertation

Convertation to `tflite` was made with help of this repo: https://github.com/AlexanderLutsenko/nobuco

### Run ResNet18

The available arguments are:

-l, --landmarks_number: Number of landmarks to detect, can be set to 29 or 98, default 98
-i, --image_path: Path to input image, default ./../Adrien_Brody.png

To test model simply run this command:

```bash
python resnet18/main.py -i /path/to/image
```

### Run Slim

The available arguments are:

-i, --image_path: Path to input image, default ./../Adrien_Brody.png

To test model simply run this command:

```bash
python slim/main.py -i /path/to/image
```

### Results

All models by default save results to `result.png` file

- slim model detects:
  - face
  - head turn: up/down-left/right
- resnet18 model with 29 landmarks number detects: 
  - face
  - head turn: up/down-left/right
- resnet18 model with 98 landmarks number detects: 
  - face
  - head turn: up/down-left/right
  - eyes status: open/close