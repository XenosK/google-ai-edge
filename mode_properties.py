import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from pathlib import Path



model_path = "models/yolo11n_float16.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)