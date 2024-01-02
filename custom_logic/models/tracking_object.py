class TrackingObject:
    def __init__(self,
                 id,
                 tracking_run_id,
                 frame_number,
                 tracking_object_id,
                 object_class,
                 center_x,
                 center_y,
                 box_width,
                 box_height,
                 speed):
        self.id = id
        self.tracking_run_id = tracking_run_id
        self.frame_number = frame_number
        self.tracking_object_id = tracking_object_id
        self.object_class = object_class
        self.center_x = center_x
        self.center_y = center_y
        self.box_width = box_width
        self.box_height = box_height
        self.speed = speed

    @property
    def bottom_y(self):
        return int(self.center_y + self.box_height / 2)

