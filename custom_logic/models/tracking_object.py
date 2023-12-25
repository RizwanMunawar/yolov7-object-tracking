class TrackingObject:
    def __init__(self,
                 TrackingRunId,
                 FrameNumber,
                 TrackingObjectId,
                 ObjectClass,
                 CenterX,
                 CenterY,
                 BoxWidth,
                 BoxHeight):
        self.BoxHeight = BoxHeight
        self.BoxWidth = BoxWidth
        self.TrackingObjectId = TrackingObjectId
        self.ObjectClass = ObjectClass
        self.CenterY = CenterY
        self.CenterX = CenterX
        self.FrameNumber = FrameNumber
        self.TrackingRunId = TrackingRunId
