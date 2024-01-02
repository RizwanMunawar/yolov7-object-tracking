from custom_logic.db.db_connection import get_db_connection
from custom_logic.models.camera import Camera


def get(id: int) -> Camera:
    with get_db_connection() as connection:
        if connection:
            cursor = connection.cursor()

            query = "SELECT * FROM cameras WHERE id = %s"
            values = (id,)

            cursor.execute(query, values)
            result = cursor.fetchone()

            cursor.close()

    if result:
        return Camera(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8])
    return None

def insert(camera: Camera):
    existing_camera = get(camera.Id)
    if existing_camera:
        return

    with get_db_connection() as connection:
        cursor = connection.cursor()

        query = "INSERT INTO cameras (manufacturer, model, focal_length, resolution_width, resolution_height, horizontal_viewing_angle, vertical_viewing_angle, sensor_size) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (camera.manufacturer, camera.model, camera.focal_length, camera.resolution_width, camera.resolution_height, camera.horizontal_viewing_angle, camera.vertical_viewing_angle, camera.sensor_size)

        cursor.execute(query, values)
        cursor.close()

