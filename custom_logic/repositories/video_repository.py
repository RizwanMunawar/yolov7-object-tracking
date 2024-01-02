from custom_logic.db.db_connection import get_db_connection
from custom_logic.models.video import Video


def get(id: int) -> Video:
    with get_db_connection() as connection:
        if connection:
            cursor = connection.cursor()

            query = "SELECT * FROM videos WHERE id = %s"
            values = (id,)

            cursor.execute(query, values)
            result = cursor.fetchone()

            cursor.close()

    if result:
        return Video(result[0], result[1], result[2], result[3], result[4], result[5], result[6],)
    return None

def get_by_title(title) -> Video:
    with get_db_connection() as connection:
        if connection:
            cursor = connection.cursor()

            query = "SELECT * FROM videos WHERE title = %s"
            values = (title,)

            cursor.execute(query, values)
            result = cursor.fetchone()

            cursor.close()

    if result:
        return Video(result[0], result[1], result[2], result[3], result[4], result[5], result[6],)
    return None


def insert(video: Video):
    existing_video = get(video.Id)
    if existing_video:
        return existing_video.Id
    existing_video = get_by_title(video.Title)
    if existing_video:
        return existing_video.Id

    with get_db_connection() as connection:
        cursor = connection.cursor()

        query = "INSERT INTO videos (title, duration, frames_number, camera_height, camera_id, tilt_angle) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (video.Title, video.Duration, video.Frames_number, video.Camera_height, video.Camera_id, video.Tilt_angle)

        cursor.execute(query, values)
        cursor.close()

    existing_video = get_by_title(video.Title)
    return existing_video.Id

