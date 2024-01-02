class SpeedEstimationConfig:
    def __init__(self, h, theta, f, horizontal_viewing_angle, vertical_viewing_angle, resolution_width, resolution_height, sensor_size):
        # Camera height
        self.H = h
        # tilt angle
        self.Theta = theta
        # focal length
        self.f = f
        self.Horizontal_viewing_angle = horizontal_viewing_angle
        self.Vertical_viewing_angle = vertical_viewing_angle
        self.Resolution_width = resolution_width
        self.Resolution_height = resolution_height
        self.Sensor_size = sensor_size
