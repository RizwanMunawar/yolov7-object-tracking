from custom_logic.db.db_connection import get_db_connection
from custom_logic.models.tracking_object import TrackingObject


def get(id: int) -> TrackingObject:
    with get_db_connection() as connection:
        if connection:
            cursor = connection.cursor()

            query = "SELECT * FROM tracking_objects WHERE id = %s"
            values = (id,)

            cursor.execute(query, values)
            result = cursor.fetchone()

            cursor.close()

    if result:
        return TrackingObject(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
                              result[8], result[9])
    return None


def get_by_tracking_run_id(tracking_run_id: int) -> list[TrackingObject]:
    with get_db_connection() as connection:
        cursor = connection.cursor()

        # Query to retrieve all tracking runs for a given video_id
        query = "SELECT * FROM tracking_objects WHERE tracking_run_id = %s"
        values = (tracking_run_id,)

        cursor.execute(query, values)
        results = cursor.fetchall()

        cursor.close()

    tracking_objects = []
    for result in results:
        tracking_object = TrackingObject(result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                                         result[7], result[8], result[9])
        tracking_objects.append(tracking_object)

    return tracking_objects


def insert(trackingObject: TrackingObject):
    with get_db_connection() as connection:
        if connection:
            # Perform database operations here
            cursor = connection.cursor()

            # Example: Execute a query
            query = get_tracking_object_insert_query(trackingObject)
            cursor.execute(query)
            cursor.close()


def set_speed(tracking_object_id: int, speed: float):
    with get_db_connection() as connection:
        if connection:
            # Perform database operations here
            cursor = connection.cursor()
            query = f"UPDATE tracking_objects SET speed = {speed} WHERE id = {tracking_object_id}"
            # Example: Execute a query
            cursor.execute(query)
            cursor.close()


def get_tracking_object_insert_query(trackingObject: TrackingObject):
    return (f"insert into tracking_objects("
            f"tracking_run_id, "
            f"frame_number, "
            f"tracking_object_id, "
            f"object_class, "
            f"center_x, "
            f"center_y, "
            f"box_width, "
            f"box_height, "
            f"speed) "
            f"values ("
            f"{trackingObject.tracking_run_id},"
            f"{trackingObject.frame_number},"
            f"{trackingObject.tracking_object_id},"
            f"{trackingObject.object_class},"
            f"{trackingObject.center_x},"
            f"{trackingObject.center_y},"
            f"{trackingObject.box_width},"
            f"{trackingObject.box_height},"
            f"null"
            f")")
