from typing import List

from custom_logic.db.db_connection import get_db_connection
from custom_logic.models.tracking_run import TrackingRun


def get(id: int) -> TrackingRun:
    with get_db_connection() as connection:
        cursor = connection.cursor()

        query = "SELECT * FROM tracking_runs WHERE id = %s"
        values = (id,)

        cursor.execute(query, values)
        result = cursor.fetchone()

        cursor.close()

    if result:
        return TrackingRun(result[0], result[1], result[2])
    return None


def get_by_video_id(video_id: int) -> List[TrackingRun]:
    with get_db_connection() as connection:
        cursor = connection.cursor()

        # Query to retrieve all tracking runs for a given video_id
        query = "SELECT * FROM tracking_runs WHERE video_id = %s"
        values = (video_id,)

        cursor.execute(query, values)
        results = cursor.fetchall()

        cursor.close()

    tracking_runs = []
    for result in results:
        tracking_run = TrackingRun(result[0], result[1], result[2])
        tracking_runs.append(tracking_run)

    return tracking_runs


def insert(trackingRun: TrackingRun):
    with get_db_connection() as connection:
        cursor = connection.cursor()

        query = "INSERT INTO tracking_runs (video_id, date) VALUES (%s, %s)"
        values = (trackingRun.VideoId, trackingRun.Date)

        cursor.execute(query, values)
        connection.commit()


def clear_tracking_run_calculated_data(tracking_run_id: int):
    with get_db_connection() as connection:
        cursor = connection.cursor()

        query = f"UPDATE tracking_objects SET speed = null, distance = null where tracking_run_id = {tracking_run_id}"

        cursor.execute(query)
        connection.commit()

