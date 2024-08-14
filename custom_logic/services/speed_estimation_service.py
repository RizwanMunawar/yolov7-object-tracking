from typing import List

import custom_logic.services.tracking_service as tracking_service
import custom_logic.repositories.tracking_object_repository as tracking_object_repository
import custom_logic.repositories.video_repository as video_repository
import custom_logic.repositories.tracking_run_repository as tracking_run_repository
import custom_logic.repositories.camera_repository as camera_repository
import custom_logic.helpers.console_helper as console_helper

from custom_logic.models.tracking_object import TrackingObject
from custom_logic.models.video import Video
from custom_logic.models.camera import Camera
from custom_logic.speed_estimator import SpeedEstimator

# frequency of speed calculations in seconds
ESTIMATION_GRANULARITY = 0.5


def estimate_tracking_run_speed(trackingRunId: int):
    tracking_run = tracking_run_repository.get(trackingRunId)
    if tracking_run is None:
        console_helper.print_error(f"Tracking run {trackingRunId} was not found.")
        return

    video = video_repository.get(tracking_run.VideoId)
    camera = camera_repository.get(video.Camera_id)
    if camera is None:
        console_helper.print_error(f"Camera is not defined for video {video.Id}. Impossible to estimate speed.")
        return

    tracking_objects = tracking_object_repository.get_by_tracking_run_id(trackingRunId)

    if tracking_objects is None or len(tracking_objects) == 0:
        console_helper.print_error(f"No objects were found for tracking run {trackingRunId}")
        return

    tracking_run_repository.clear_tracking_run_calculated_data(trackingRunId)

    fps = video.Frames_number / video.Duration
    # how often estimation will be done
    number_of_frames_between_estimations = int(fps * ESTIMATION_GRANULARITY)
    # since number of frames is integer then we'll not have exact ESTIMATION_GRANULARITY. It should be float to be more precise
    real_granularity = number_of_frames_between_estimations / fps

    speed_estimator = SpeedEstimator(video.Title, real_granularity)

    frames = tracking_service.group_tracking_objects(tracking_objects)
    previous_frame = frames.get(1)

    for frame in range(2, video.Frames_number + 1):
        if (frame - 1) % number_of_frames_between_estimations != 0:
            continue

        current_frame = frames.get(frame)
        if current_frame is None:
            previous_frame = None
            continue

        print(f"Estimating frame {frame}. ({previous_frame[0].center_x}, {previous_frame[0].center_y}) - ({current_frame[0].center_x}, {current_frame[0].center_y})")

        estimate_frames(previous_frame, current_frame, speed_estimator)
        previous_frame = current_frame


def estimate_frames(previous_frame: List[TrackingObject], current_frame: List[TrackingObject], speed_estimator: SpeedEstimator):
    for tracking_object in current_frame:
        filtered = filter(lambda x: x.tracking_object_id == tracking_object.tracking_object_id, previous_frame)
        previous_tracking_object = next(filtered, None)

        if previous_tracking_object is None:
            continue

        speed, distance = speed_estimator.get_estimated_speed_and_distance(previous_tracking_object, tracking_object)
        tracking_object_repository.set_speed_and_distance(tracking_object.id, speed, distance)
