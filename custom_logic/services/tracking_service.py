from typing import List

import custom_logic.repositories.video_repository as video_repository
import custom_logic.repositories.tracking_run_repository as tracking_run_repository
import custom_logic.repositories.tracking_object_repository as tracking_object_repository

from custom_logic.models.tracking_object import TrackingObject
from custom_logic.models.tracking_run import TrackingRun
from datetime import datetime


def insert_tracking_run(videoId: int):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tracking_run_repository.insert(TrackingRun(None, videoId, current_datetime))

    return get_last_tracking_run_id(videoId)


def get_last_tracking_run_id(video_id: int):
    return tracking_run_repository.get_by_video_id(video_id)[-1].Id


def get_tracking_objects_for_video(name, tracking_run_id=None):
    if tracking_run_id is None:
        video = video_repository.get_by_title(name)
        if video is None:
            return []

        tracking_run_id = get_last_tracking_run_id(video.Id)

    return tracking_object_repository.get_by_tracking_run_id(tracking_run_id)


def group_tracking_objects(tracking_objects: List[TrackingObject]):
    return {key: [item for item in tracking_objects if item.frame_number == key] for key in
            set(map(lambda item: item.frame_number, tracking_objects))}
