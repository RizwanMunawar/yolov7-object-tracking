## YOLOv7 Object Tracking 🚀

### New Features ✨
- 🏷️ Added Label for Every Track
- ⚡ Runs on both CPU & GPU
- 🎥 Supports Video, Webcam, External Camera, and IP Stream

### Coming Soon 🔄
- 🛠️ Development of a Streamlit Dashboard for Object Tracking

### Ready-to-Use Google Colab 🔗 [Launch Colab](https://colab.research.google.com/drive/1xrB76UQ_LaVaBAxfTi8-a9dIcazmxD5b?usp=sharing)

### How to Run the Code 🖥️
1. **Clone the repository:**
    ```bash
    git clone https://github.com/RizwanMunawar/yolov7-object-tracking.git
    ```
2. **Navigate to the cloned folder:**
    ```bash
    cd yolov7-object-tracking
    ```

3. **Create a virtual environment (Recommended to avoid conflicts):**

    #### For Anaconda:
    ```bash
    conda create -n yolov7objtracking python=3.10
    conda activate yolov7objtracking
    ```

    #### For Linux:
    ```bash
    python3 -m venv yolov7objtracking
    source yolov7objtracking/bin/activate
    ```

    #### For Windows:
    ```bash
    python3 -m venv yolov7objtracking
    cd yolov7objtracking/Scripts
    activate
    ```

4. **Update pip and install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5. **Run the script:**

    Select the appropriate command based on your requirements. Pretrained [yolov7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) weights will be downloaded automatically if needed.

    - Detection only:
      ```bash
      python detect.py --weights yolov7.pt --source "your video.mp4"
      ```

    - Object tracking:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4"
      ```

    - Webcam:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source 0
      ```

    - External Camera:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source 1
      ```

    - IP Camera Stream:
      ```bash
      python detect_and_track.py --source "your IP Camera Stream URL" --device 0
      ```

    - Specific class tracking (e.g., person):
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --classes 0
      ```

    - Colored tracks:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --colored-trk
      ```

    - Save track centroids, IDs, and bounding box coordinates:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --save-txt --save-bbox-dim
      ```

6. **Output files** will be saved in `working-dir/runs/detect/obj-tracking` with the original filename.

### Results 📊
<table>
  <tr>
    <td>YOLOv7 Detection Only</td>
    <td>YOLOv7 Object Tracking with ID</td>
    <td>YOLOv7 Object Tracking with ID and Label</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/196107891-bb8124de-99c6-4039-b556-2ade403bd985.png"></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185798283-0455ce49-4359-4e52-8d69-fd30dd61c5b4.png"></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/191241661-ed5b87eb-5c8c-49bc-8301-531ee86f3b38.png"></td>
  </tr>
</table>

### References 🔗
- [YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)
- [SORT GitHub](https://github.com/abewley/sort)

### My Medium Articles ✍️
- [Maximizing Efficiency on Construction Sites 🔥](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/04/maximizing-efficiency-on-construction.html)
- [Instance Segmentation vs Semantic Segmentation ✅](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/instance-segmentation-vs-semantic.html)
- [Mastering Image Classification 🔥](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/mastering-image-classification.html)
- [Object Detection in Agriculture ✅](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/object-detection-in-agriculture.html)
- [Techniques for Accurate Data Annotation ✅](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/techniques-for-accurate-data-annotation.html)
- [Object Tracking Using ByteTrack 🔥](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/object-tracking-using-bytetrack.html)
- [Pose Estimation in Computer Vision ✅](https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/pose-estimation-computer-vision.html)

For more details, you can reach me on [Medium](https://muhammadrizwanmunawar.medium.com/) or [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/).
