import cv2
from typing import List
from custom_logic.models.tracking_object import TrackingObject
from custom_logic.tracking_repository import get_last_tracking_objects_for_video


def drawFrameBoxes(img, trackingObjects: List[TrackingObject]):
    for item in trackingObjects:
        drawBox(img, item)


def drawBox(img, tracking_object: TrackingObject):
    x1 = int(tracking_object.CenterX - tracking_object.BoxWidth / 2)
    x2 = int(tracking_object.CenterX + tracking_object.BoxWidth / 2)
    y1 = int(tracking_object.CenterY - tracking_object.BoxHeight / 2)
    y2 = int(tracking_object.CenterY + tracking_object.BoxHeight / 2)
    label = str(tracking_object.TrackingObjectId)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, [255, 0, 20], 1)


source = "shopping-mall-people-walking.mp4"
run_id = None

capture = cv2.VideoCapture(source)

tracking_objects = get_last_tracking_objects_for_video(source, run_id)
tracking_objects_by_frame = {key: [item for item in tracking_objects if item.FrameNumber == key] for key in
                             set(map(lambda item: item.FrameNumber, tracking_objects))}
frame = 1

while capture.isOpened():
    _, img_1 = capture.read()
    if img_1 is None:
        continue

    frame_tracking_objects = tracking_objects_by_frame.get(frame)
    drawFrameBoxes(img_1, frame_tracking_objects if frame_tracking_objects is not None else [])
    frame += 1

    cv2.imshow("Detecting Motion...", img_1)
    if cv2.waitKey(30) == 13:
        exit()
