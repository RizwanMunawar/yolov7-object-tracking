from custom_logic.models.tracking_object import TrackingObject
from custom_logic.distance_measurer import DistanceMeasurer
from custom_logic.models.point import Point


def get_tracking_point(tracking_object: TrackingObject) -> Point:
    return Point(tracking_object.center_x, int(tracking_object.center_y + tracking_object.box_height / 2))

class SpeedEstimator:
    def __init__(self, video_title: str, estimation_granularity: float):
        self.Distance_measurer = DistanceMeasurer(video_title)
        self.Estimation_granularity = estimation_granularity


    def get_estimated_speed_and_distance(self, first_tracking_object: TrackingObject,
                                         second_tracking_object: TrackingObject) -> float:
        p1 = get_tracking_point(first_tracking_object)
        p2 = get_tracking_point(second_tracking_object)
        distance = self.Distance_measurer.get_distance(p1, p2)
        speed = distance / self.Estimation_granularity
        print(f"Object {first_tracking_object.tracking_object_id} speed - {speed} m/s")
        return speed, distance
