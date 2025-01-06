import onnx
import onnxruntime as ort
import cv2
from PIL import Image
import numpy as np
import os
from collections import Counter

model = onnx.load('best_video_simplified3.onnx')
onnx.checker.check_model(model)
 
session = ort.InferenceSession('best_video_simplified3.onnx')

path = "./test_dataset/"

count_all = 0
right_all = 0

classify_right = [0,0,0,0,0,0,0]
classify_all = [0,0,0,0,0,0,0]
precision_classify = [0,0,0,0,0,0,0]
gt_class = ["grooming","head_raising","normal","rearing","genital_licking","wall_supported_rearing","face_washing"]
for class_name in os.listdir(path):
    print(class_name)
    gt = -1
    if(class_name == "grooming"):
      gt = 0
    elif(class_name == "head_raising"):
      gt = 1
    elif(class_name == "normal"):
      gt = 2
    elif(class_name == "rearing"):
      gt = 3
    elif(class_name == "genital_licking"):
      gt = 4
    elif(class_name == "wall_supported_rearing"):
      gt = 5
    elif(class_name == "face_washing"):
      gt = 6
   
    count = 0
    right = 0
    class_path = os.path.join(path, class_name)
    for video_name in os.listdir(class_path):
      video_path = os.path.join(class_path, video_name)
      
      pr = []
      cap = cv2.VideoCapture(video_path)
      rval, frame = cap.read()
      while (rval):
        rval, frame = cap.read()
        if rval==True:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = cv2.resize(frame, dsize=(640,640))
          frame = Image.fromarray(np.uint8(frame))
          frame = np.array(frame)
          frame = np.transpose(frame, (2, 0, 1))
          frame = frame.astype(np.float32)
          frame = (frame - 127.5) / 127.5
          frame = frame[None]

          outputs = session.run(None,{ 'input' : frame})
          my_list = outputs[0][0]
          pr.append(np.argmax(my_list))
      
      counter = Counter(pr)
      most_common_value, most_common_count = counter.most_common(1)[0]
      
      count+=1
      count_all+=1
      classify_all[most_common_value]+=1
      if(most_common_value == gt):
        right+=1
        right_all+=1
        classify_right[most_common_value]+=1
    print(class_name, "-precision: ", float(right)/float(count))
    precision_classify[gt] =  float(right)/float(count)
print("top1: ", float(right_all)/float(count_all))

for i in range(0,7):
  print(gt_class[i], "-recall: ", float(classify_right[i])/float(classify_all[i]))

for i in range(0,7):
  print(gt_class[i], "-f1: ", float(2.0)/(1/float(precision_classify[i])+1/(float(classify_right[i])/float(classify_all[i]))))