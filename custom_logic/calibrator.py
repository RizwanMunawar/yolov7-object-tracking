import math

from custom_logic.distance_measurer import DistanceMeasurer
from custom_logic.distance_meter import DistanceMeter
from custom_logic.models.point import Point
import numpy as np


class CalibratingParameter:
    def __init__(self, disperse: float, step: float):
        self.disperse = disperse
        self.step = step


def print_values(tilt_angle, height, focal_length, sensor_size, loose):
    print(f"Best parameters:\n"
          f"Tilt angle = {tilt_angle}\n"
          f"Height = {height}\n"
          f"Focal length = {focal_length}\n"
          f"Sensor size = {sensor_size}\n"
          f"Loose = {loose}\n")


class Calibrator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        video_title = video_path.split("/")[-1]
        self.Distance_measurer = DistanceMeasurer(video_title)
        self.Test_paths = [(Point(1658, 686), Point(928, 384), 11.25),
                           (Point(651, 490), Point(1658, 685), 7.35),
                           (Point(528, 704), Point(722, 379), 11.1),
                           (Point(743, 373), Point(898, 371), 1.8),
                           (Point(746, 372), Point(785, 298), 6.1),
                           (Point(906, 364), Point(911, 302), 6.1),
                           (Point(1221, 174), Point(1535, 173), 10.6),
                           (Point(1054, 160), Point(1104, 160), 2)]
        self.TILT_ANGLE = CalibratingParameter(3, 0.1)
        self.HEIGHT = CalibratingParameter(10, 1)
        self.FOCAL_LENGTH = CalibratingParameter(1.5, 0.1)
        self.SENSOR_SIZE = CalibratingParameter(3, 0.1)

    def run(self):
        best_tilt_angle, best_height, best_focal_length, best_sensor_size = self.get_current_values()
        best_loose = 10000000
        step = 1

        initial_loose = 0
        for test_path in self.Test_paths:
            distance = self.Distance_measurer.get_distance(test_path[0], test_path[1])
            initial_loose = initial_loose + (distance - test_path[2]) ** 2
            
        initial_theta = self.Distance_measurer.Config.Theta
        initial_H = self.Distance_measurer.Config.H
        initial_sensor_size = self.Distance_measurer.Config.Sensor_size
        initial_f = self.Distance_measurer.Config.f

        for tilt_angle in np.arange(initial_theta - self.TILT_ANGLE.disperse,
                                    initial_theta + self.TILT_ANGLE.disperse,
                                    self.TILT_ANGLE.step):
            for height in np.arange(initial_H - self.HEIGHT.disperse,
                                    initial_H + self.HEIGHT.disperse,
                                    self.HEIGHT.step):
                for focal_length in np.arange(initial_f - self.FOCAL_LENGTH.disperse,
                                              initial_f + self.FOCAL_LENGTH.disperse,
                                              self.FOCAL_LENGTH.step):
                    for sensor_size in np.arange(initial_sensor_size - self.SENSOR_SIZE.disperse,
                                                 initial_sensor_size + self.SENSOR_SIZE.disperse,
                                                 self.SENSOR_SIZE.step):
                        if step % 100000 == 0:
                            print(f"Step - {step}")
                        step = step + 1
                        self.Distance_measurer.Config.Theta = tilt_angle
                        self.Distance_measurer.Config.H = height
                        self.Distance_measurer.Config.f = focal_length
                        self.Distance_measurer.Config.Sensor_size = sensor_size
                        loose = 0
                        for test_path in self.Test_paths:
                            distance = self.Distance_measurer.get_distance(test_path[0], test_path[1])
                            loose = loose + (distance - test_path[2]) ** 2
                        loose = math.sqrt(loose)
                        if loose < best_loose:
                            best_loose = loose
                            best_tilt_angle, best_height, best_focal_length, best_sensor_size = self.get_current_values()
                            print(f"New best on step {step}")
                            print_values(best_tilt_angle, best_height, best_focal_length, best_sensor_size, best_loose)

        print_values(best_tilt_angle, best_height, best_focal_length, best_sensor_size, best_loose)
        print(f"Initial loose - {initial_loose}")
        distance_meter = DistanceMeter(self.video_path)
        self.Distance_measurer.Config.Theta = best_tilt_angle
        self.Distance_measurer.Config.H = best_height
        self.Distance_measurer.Config.f = best_focal_length
        self.Distance_measurer.Config.Sensor_size = best_sensor_size
        distance_meter.Distance_measurer = self.Distance_measurer
        distance_meter.start()

    def get_current_values(self):
        return self.Distance_measurer.Config.Theta, self.Distance_measurer.Config.H, self.Distance_measurer.Config.f, self.Distance_measurer.Config.Sensor_size
