import argparse
from custom_logic.services.speed_estimation_service import estimate_tracking_run_speed
from custom_logic.video_displayer import VideoDisplayer
from custom_logic.distance_meter import DistanceMeter
from custom_logic.calibrator import Calibrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processor of custom actions.")
    parser.add_argument("--action", type=str, help="The action to do")
    parser.add_argument("--tracking-run-id", type=int, help="TrackingRunId. Required for estimate-tracking-run-speed action")
    parser.add_argument("--video-path", type=str, help="Video path")
    parser.add_argument("--save-video", type=bool, help="Is save video")
    parser.add_argument("--draw-track", type=bool, help="Is draw tracks")

    args = parser.parse_args()

    # Check if required arguments are provided
    if not all([args.action]):
        parser.print_help()
    elif args.action == 'estimate-tracking-run-speed':
        if args.tracking_run_id is None or args.tracking_run_id <= 0:
            print("tracking-run-id is required")
        else:
            estimate_tracking_run_speed(args.tracking_run_id)
    elif args.action == "display-video":
        if args.video_path is None and args.tracking_run_id is None:
            print("tracking-run-id and video-path is required")
        else:
            is_save_video = args.save_video
            if is_save_video is None:
                is_save_video = False

            is_draw_track = args.draw_track
            if is_draw_track is None:
                is_draw_track = False

            video_displayer = VideoDisplayer(args.video_path, args.tracking_run_id, is_save_video, is_draw_track)
            video_displayer.display_video()
    elif args.action == "distance-meter":
        if args.video_path is None:
            print("video-path is required")
        else:
            distance_meter = DistanceMeter(args.video_path, args.tracking_run_id)
            distance_meter.start()
    elif args.action == "calibrate":
        if args.video_path is None:
            print("video-path is required")
        else:
            calibrator = Calibrator(args.video_path)
            calibrator.run()
    elif args.action == "custom":
        video_path = "./video_sources/Test_video_3_correct.mp4"
        tracking_run_id = 12

        # Estimate speed
        # estimate_tracking_run_speed(tracking_run_id)

        # Show video
        is_save_video = False
        video_displayer = VideoDisplayer(video_path, tracking_run_id, is_save_video)
        video_displayer.display_video()
else:
        print("Command not found")
