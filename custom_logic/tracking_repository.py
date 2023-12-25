from custom_logic.db_connection import MySQLConnection
from custom_logic.credentials import ConnectionConstants
from custom_logic.models.tracking_object import TrackingObject
from custom_logic.models.video import Video
from custom_logic.models.tracking_run import TrackingRun
from datetime import datetime


def get_db_connection():
    return MySQLConnection(
        ConnectionConstants.HOST,
        ConnectionConstants.USERNAME,
        ConnectionConstants.PASSWORD,
        ConnectionConstants.DATABASE)


def add_video(video: Video):
    existing_video = get_video(video)
    if existing_video:
        return existing_video.Id

    with get_db_connection() as connection:
        cursor = connection.cursor()

        query = "INSERT INTO video (Name, Duration, FramesNumber) VALUES (%s, %s, %s)"
        values = (video.Name, video.Duration, video.FramesNumber)

        cursor.execute(query, values)
        cursor.close()

    existing_video = get_video(video)
    return existing_video.Id


def insert_tracking_run(videoId: int):
    with get_db_connection() as connection:
        cursor = connection.cursor()
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = "INSERT INTO tracking_run (VideoId, Date) VALUES (%s, %s)"
        values = (videoId, current_datetime)

        cursor.execute(query, values)
        cursor.close()

    return get_last_tracking_run_id(videoId)


def track_object_frame_data(trackingObject: TrackingObject):
    with get_db_connection() as connection:
        if connection:
            # Perform database operations here
            cursor = connection.cursor()

            # Example: Execute a query
            cursor.execute(get_tracking_object_insert_query(trackingObject))

            connection.commit()


def get_video(video: Video):
    with get_db_connection() as connection:
        if connection:
            cursor = connection.cursor()

            query = "SELECT * FROM videos WHERE Id = %s OR Name = %s"
            values = (video.Id, video.Name)

            cursor.execute(query, values)
            result = cursor.fetchone()

            cursor.close()

    if result:
        return Video(result[0], result[1], result[2], result[3])
    return None


def get_last_tracking_run_id(video_id: int):
    return get_tracking_runs_by_video_id(video_id)[-1].Id


def get_tracking_runs_by_video_id(video_id: int):
    with get_db_connection() as connection:
        cursor = connection.cursor()

        # Query to retrieve all tracking runs for a given video_id
        query = "SELECT * FROM tracking_runs WHERE VideoId = %s"
        values = (video_id,)

        cursor.execute(query, values)
        results = cursor.fetchall()

        cursor.close()

    tracking_runs = []
    for result in results:
        tracking_run = TrackingRun(result[0], result[1], result[2])
        tracking_runs.append(tracking_run)

    return tracking_runs


def get_last_tracking_objects_for_video(name, tracking_run_id=None):
    if tracking_run_id is None:
        video = get_video(Video(None, name, None, None))
        if video is None:
            return []

        tracking_run_id = get_last_tracking_run_id(video.Id)

    with get_db_connection() as connection:
        cursor = connection.cursor()

        # Query to retrieve all tracking runs for a given video_id
        query = "SELECT * FROM tracking_objects WHERE TrackingRunId = %s"
        values = (tracking_run_id,)

        cursor.execute(query, values)
        results = cursor.fetchall()

        cursor.close()

    tracking_objects = []
    for result in results:
        tracking_object = TrackingObject(result[1], result[2], result[3], result[4], result[5], result[6], result[7],
                                         result[8])
        tracking_objects.append(tracking_object)

    return tracking_objects


def get_tracking_object_insert_query(trackingObject: TrackingObject):
    return (f"insert into tracking_objects("
            f"TrackingRunId, "
            f"FrameNumber, "
            f"TrackingObjectId, "
            f"ObjectClass, "
            f"CenterX, "
            f"CenterY, "
            f"BoxWidth, "
            f"BoxHeight) "
            f"values ("
            f"{trackingObject.TrackingRunId},"
            f"{trackingObject.FrameNumber},"
            f"{trackingObject.TrackingObjectId},"
            f"{trackingObject.ObjectClass},"
            f"{trackingObject.CenterX},"
            f"{trackingObject.CenterY},"
            f"{trackingObject.BoxWidth},"
            f"{trackingObject.BoxHeight})");
