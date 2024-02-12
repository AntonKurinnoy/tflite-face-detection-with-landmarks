## Example of using `tflite` model to detect faces with landmarks

### General
This is the simple usage of `tflite` model to detect faces with landmarks.
In addition, a simple detection of head turn has been added.

### Original model
Slim model was taken from this repo: https://github.com/biubug6/Face-Detector-1MB-with-landmark

### Convertation
Convertation to `tflite` was made with help of this repo: https://github.com/AlexanderLutsenko/nobuco

### Run
To test model simply run this command:

```bash
python main.py -i Adrien_Brody.png
```
