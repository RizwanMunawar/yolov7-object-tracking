## YOLOv7 Object Tracking 🚀

<a href="https://colab.research.google.com/drive/1xrB76UQ_LaVaBAxfTi8-a9dIcazmxD5b?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://deepwiki.com/RizwanMunawar/yolov7-object-tracking"><img src="https://img.shields.io/badge/DeepWiki-RizwanMunawar%2Fyolov7_object_tracking-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="YOLOv7-object-tracking DeepWiki"></a>

### How to Run the Code

1. Clone the repository:
   
    ```bash
    git clone https://github.com/RizwanMunawar/yolov7-object-tracking.git
    ```
   
2. Navigate to the cloned folder:
    ```bash
    cd yolov7-object-tracking
    ```

3. Create a virtual environment (Recommended to avoid conflicts):

    **For Anaconda**:
    ```bash
    conda create -n yolov7objtracking python=3.10
    conda activate yolov7objtracking
    ```

    **For Linux**:
    ```bash
    python3 -m venv yolov7objtracking
    source yolov7objtracking/bin/activate
    ```

    **For Windows**:
    ```bash
    python3 -m venv yolov7objtracking
    cd yolov7objtracking/Scripts
    activate
    ```

4. Update pip and install dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5. Run the script:

    Select the appropriate command based on your requirements. Pretrained [yolov7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) weights will be downloaded automatically if needed.

    - **Detection only**:
      ```bash
      python detect.py --weights yolov7.pt
  
      # If you want to use your own videos.
      python detect.py --weights yolov7.pt --source "your video.mp4"
      ```

    - **Object tracking**:
      ```bash
      python detect_and_track.py --weights yolov7.pt
      
      # If you want to use your own videos.
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4"
      ```

    - **Webcam**:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source 0
      ```

    - **External Camera**:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source 1
      ```

    - **IP Camera Stream**:
      ```bash
      python detect_and_track.py --source "your IP Camera Stream URL" --device 0
      ```

    - **Specific class tracking (e.g., person)**:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --classes 0
      ```

    - **Colored tracks**:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --colored-trk
      ```

    - **Save track centroids, IDs, and bounding box coordinates**:
      ```bash
      python detect_and_track.py --weights yolov7.pt --source "your video.mp4" --save-txt --save-bbox-dim
      ```

6. **Output files** will be saved in `working-dir/runs/detect/obj-tracking` with the original filename.

### Arguments details 🚀

| Argument                | Type        | Default                       | Description                                                         |
|-------------------------|-------------|-------------------------------|---------------------------------------------------------------------|
| `--weights`             | `str`       | `yolov7.pt`                   | Path(s) to model weights (`.pt` file).                              |
| `--download`            | `flag`      | `False`                       | Download model weights automatically.                               |
| `--no-download`         | `flag`      | `False`                       | Do not download model weights if they already exist.                |
| `--source`              | `str`       | `None`                        | Source for inference (file, folder, or `0` for webcam).             |
| `--img-size`            | `int`       | `640`                         | Inference image size in pixels.                                     |
| `--conf-thres`          | `float`     | `0.25`                        | Object confidence threshold.                                        |
| `--iou-thres`           | `float`     | `0.45`                        | Intersection over Union (IoU) threshold for NMS.                    |
| `--device`              | `str`       | `''`                          | CUDA device (e.g., `0` or `0,1,2,3`) or `cpu`.                      |
| `--view-img`            | `flag`      | `False`                       | Display results during inference.                                   |
| `--save-txt`            | `flag`      | `False`                       | Save results to `.txt` files.                                       |
| `--save-conf`           | `flag`      | `False`                       | Save confidence scores in `.txt` labels.                            |
| `--nosave`              | `flag`      | `False`                       | Do not save images or videos.                                       |
| `--classes`             | `list[int]` | `None`                        | Filter results by class (e.g., `--classes 0` or `--classes 0 2 3`). |
| `--agnostic-nms`        | `flag`      | `False`                       | Use class-agnostic Non-Maximum Suppression (NMS).                   |
| `--augment`             | `flag`      | `False`                       | Enable augmented inference.                                         |
| `--update`              | `flag`      | `False`                       | Update all models.                                                  |
| `--project`             | `str`       | `runs/detect`                 | Directory to save results (`project/name`).                         |
| `--name`                | `str`       | `runs/detect/object_tracking` | Name of the results folder inside the project directory.            |
| `--exist-ok`            | `flag`      | `False`                       | Allow existing project/name without incrementing.                   |
| `--no-trace`            | `flag`      | `False`                       | Do not trace the model during export.                               |
| `--colored-trk`         | `flag`      | `False`                       | Assign a unique color to each track for visualization.              |
| `--save-bbox-dim`       | `flag`      | `False`                       | Save bounding box dimensions in `.txt` tracks.                      |
| `--save-with-object-id` | `flag`      | `False`                       | Save results with object ID in `.txt` files.                        |

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

### References

- [YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)
- [SORT GitHub](https://github.com/abewley/sort)

**Some of my articles/research papers | computer vision awesome resources for learning | How do I appear to the world? 🚀**

[Ultralytics YOLO11: Object Detection and Instance Segmentation🤯](https://muhammadrizwanmunawar.medium.com/ultralytics-yolo11-object-detection-and-instance-segmentation-88ef0239a811) ![Published Date](https://img.shields.io/badge/published_Date-2024--10--27-brightgreen)

[Parking Management using Ultralytics YOLO11](https://muhammadrizwanmunawar.medium.com/parking-management-using-ultralytics-yolo11-fba4c6bc62bc) ![Published Date](https://img.shields.io/badge/published_Date-2024--11--10-brightgreen)

[My 🖐️Computer Vision Hobby Projects that Yielded Earnings](https://muhammadrizwanmunawar.medium.com/my-️computer-vision-hobby-projects-that-yielded-earnings-7923c9b9eead) ![Published Date](https://img.shields.io/badge/published_Date-2023--09--10-brightgreen)

[Best Resources to Learn Computer Vision](https://muhammadrizwanmunawar.medium.com/best-resources-to-learn-computer-vision-311352ed0833) ![Published Date](https://img.shields.io/badge/published_Date-2023--06--30-brightgreen)

[Roadmap for Computer Vision Engineer](https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c)  ![Published Date](https://img.shields.io/badge/published_Date-2022--08--07-brightgreen)

[How did I spend 2022 in the Computer Vision Field](https://www.linkedin.com/pulse/how-did-i-spend-2022-computer-vision-field-muhammad-rizwan-munawar) ![Published Date](https://img.shields.io/badge/published_Date-2022--12--20-brightgreen)

[Domain Feature Mapping with YOLOv7 for Automated Edge-Based Pallet Racking Inspections](https://www.mdpi.com/1424-8220/22/18/6927) ![Published Date](https://img.shields.io/badge/published_Date-2022--09--13-brightgreen)

[Exudate Regeneration for Automated Exudate Detection in Retinal Fundus Images](https://ieeexplore.ieee.org/document/9885192) ![Published Date](https://img.shields.io/badge/published_Date-2022--09--12-brightgreen)

[Feature Mapping for Rice Leaf Defect Detection Based on a Custom Convolutional Architecture](https://www.mdpi.com/2304-8158/11/23/3914) ![Published Date](https://img.shields.io/badge/published_Date-2022--12--04-brightgreen)

[Yolov5, Yolo-x, Yolo-r, Yolov7 Performance Comparison: A Survey](https://aircconline.com/csit/papers/vol12/csit121602.pdf)  ![Published Date](https://img.shields.io/badge/published_Date-2022--09--24-brightgreen)

[Explainable AI in Drug Sensitivity Prediction on Cancer Cell Lines](https://ieeexplore.ieee.org/document/9922931)  ![Published Date](https://img.shields.io/badge/published_Date-2022--09--23-brightgreen)

[Train YOLOv8 on Custom Data](https://medium.com/augmented-startups/train-yolov8-on-custom-data-6d28cd348262)  ![Published Date](https://img.shields.io/badge/published_Date-2022--09--23-brightgreen)

[Session with Ultralytics Team about Computer Vision Journey](https://www.ultralytics.com/blog/becoming-a-computer-vision-engineer)  ![Published Date](https://img.shields.io/badge/published_Date-2022--11--15-brightgreen)
