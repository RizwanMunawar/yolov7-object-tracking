# yolov7-object-tracking

### New Features
- Added Label for Every Track
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Development of streamlit dashboard for Object Tracking

### Ready to Use Google Colab
- https://colab.research.google.com/drive/1xrB76UQ_LaVaBAxfTi8-a9dIcazmxD5b?usp=sharing
### Steps to run Code
- Clone the repository.
```
git clone https://github.com/RizwanMunawar/yolov7-object-tracking.git
```
- Goto the cloned folder.
```
cd yolov7-object-tracking
```
- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv yolov7objtracking
source yolov7objtracking/bin/activate

### For Window Users
python3 -m venv yolov7objtracking
cd yolov7objtracking
cd Scripts
activate
cd ..
cd ..
```
- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Run the code with mentioned command below (by default, pretrained [yolov7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) weights will be automatically downloaded into the working directory if they don't already exist).
```
# for detection only
python detect.py --weights yolov7.pt --source "your video.mp4"

#if you want to change source file
python detect_and_track.py --weights yolov7.pt --source "your video.mp4"

#for WebCam
python detect_and_track.py --weights yolov7.pt --source 0

#for External Camera
python detect_and_track.py --weights yolov7.pt --source 1

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python detect_and_track.py --source "your IP Camera Stream URL" --device 0

#for specific class (person)
python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --classes 0

#for colored tracks 
python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --colored-trk

#for saving tracks centroid, track id and bbox coordinates
python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --save-txt --save-bbox-dim
```

- Output file will be created in the ```working-dir/runs/detect/obj-tracking``` with original filename


### Results
<table>
  <tr>
    <td>YOLOv7 Detection Only</td>
    <td>YOLOv7 Object Tracking with ID</td>
    <td>YOLOv7 Object Tracking with ID and Label </td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/196107891-bb8124de-99c6-4039-b556-2ade403bd985.png"></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185798283-0455ce49-4359-4e52-8d69-fd30dd61c5b4.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/191241661-ed5b87eb-5c8c-49bc-8301-531ee86f3b38.png"></td>
  </tr>
 </table>


 ### References
 - https://github.com/WongKinYiu/yolov7
 - https://github.com/abewley/sort

### My Medium Articles
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/04/maximizing-efficiency-on-construction.html 🔥
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/instance-segmentation-vs-semantic.html ✅
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/mastering-image-classification.html 🔥
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/object-detection-in-agriculture.html ✅
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/techniques-for-accurate-data-annotation.html ✅
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/object-tracking-using-bytetrack.html 🔥
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/pose-estimation-computer-vision.html ✅
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f

For more details, you can reach out to me on [Medium](https://chr043416.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)
