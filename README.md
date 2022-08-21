# yolov7-object-tracking

### New Features
- Added Label for Every Track
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Development of streamlit dashboard for Object Tracking and Counting

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
- Download [yolov7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) object detection weights from link and move them to the working directory {yolov7-object-tracking}
- Run the code with mentioned command below.
```
#if you want to change source file
python detect_and_track.py --weights yolov7.pt --source "your video.mp4"

#for specific class (person)
python detect_and_track.py --weights yolov7.pt --source "your video.mp4" -classes 0
```
- Output file will be created in the ```working-dir/runs/detect/obj-tracking``` with original filename

### Results
<table>
  <tr>
    <td>YOLOv7 Object Tracking</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185798283-0455ce49-4359-4e52-8d69-fd30dd61c5b4.png"></td>
  </tr>
 </table>
 
 ### References
 - https://github.com/WongKinYiu/yolov7
 - https://github.com/abewley/sort

### My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f

For more details, you can reach out to me on [Medium](https://chr043416.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)
