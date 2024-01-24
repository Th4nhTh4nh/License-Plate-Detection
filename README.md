# Vietnamese-License-Plate-Recognition-YOLOv3-and-CNN
## 1. Introduction
This is my personal project, I did it while studying at school. The program performs tasks including object detection using YOLOv3 and character recognition using CNN.<br>
There are 3 main stages in my license-plate-recoginition project:<br>
<p align="center">
  <img width="284" alt="aa" src="https://github.com/Th4nhTh4nh/License-Plate-Detection/assets/111641722/766da0d2-6129-448f-86f8-65f90fc9df98"> <br>
  <i>3 main stages</i>
</p>

## 2. License Plate Detection
I use YOLO v3 to detect license plates in photos because it is convenient and highly accurate. All I have to do is collecting the data and train the model.<br>
* The license plate database I use is an image database consisting of 940 photos of both 1 and 2 lines Vietnamese license plates. You can find whole dataset in here [dataset](https://thigiacmaytinh.com/tai-nguyen-xu-ly-anh/tong-hop-data-xu-ly-anh/?fbclid=IwAR2tajA5Ku83kIrb09ovhmb_68Zmdwo9KvV_CSNBCTbuIIsiK_FUM4W4Dh8.)
* I used [labelImg](https://github.com/HumanSignal/labelImg#create-pre-defined-classes) to label each images. I will have the .txt file in the same folder with the image. .txt file include label, x, y, w, h. <br>
The detection result is quite good:
<p align="center">
  <img width="279" alt="11" src="https://github.com/Th4nhTh4nh/License-Plate-Detection/assets/111641722/0eb9e043-c957-4fb0-a966-8326a4c2077c"> <br>
  <i>The license plate was detected.</i><br>
  <br>
  <img width="393" alt="12" src="https://github.com/Th4nhTh4nh/License-Plate-Detection/assets/111641722/143e986e-4ef6-4db6-b2ca-5eb79b9cccb6"><br>
  <i>Crop the image.</i>
</p>
After being cropped, the gray image may still have noise, caused by license plate edges and possibly dirt on the license plate, so we need to pre-process the image to filter and remove noise from the image.
<p align="center">
  <img width="290" alt="13" src="https://github.com/Th4nhTh4nh/License-Plate-Detection/assets/111641722/ffb3cd54-d9b7-483e-aa58-7334a690bb73"><br>
  <i>The image has been filtered for noise.</i>
</p>

## 3. Character Segmentation
I use the method of finding boundaries from the image of the license plate after being processed by thresholding. <br>Each border found can correspond to a character on the license plate.<br>So, the tasks to do in this section are:
* Find contours.
* Recognize with CNN.
<p align="center">
  <img width="401" alt="14" src="https://github.com/Th4nhTh4nh/License-Plate-Detection/assets/111641722/cbe65503-cac4-466f-b69e-81057011c3bc"><br>
  <i>Characters are separated into individual images.</i>
</p>

## 4. Recognize License Plate
I built a simple convolutional neural network architecture that is responsible for classifying images of capital letters (From A to Z) and numeric characters (From 0 to 9).<br>I built a simple Flask app to demo my program.
<p align="center">
  <img width="399" alt="15" src="https://github.com/Th4nhTh4nh/License-Plate-Detection/assets/111641722/d7078403-5327-407e-a258-0d8561a74397"><br>
  <i>Final result.</i>
</p>

