import cv2


def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the duration of the video in seconds
    duration_seconds = total_frames / fps

    # Release the video capture object
    cap.release()

    return total_frames, fps, duration_seconds

# Example usage
video_path = 'path/to/your/video.mp4'
video_info = get_video_info(video_path)

if video_info:
    total_frames, fps, duration_seconds = video_info
    print(f"Total Frames: {total_frames}")
    print(f"Frames Per Second (fps): {fps}")
    print(f"Duration (seconds): {duration_seconds}")
