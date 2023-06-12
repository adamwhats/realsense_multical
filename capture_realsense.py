import logging
import os
import shutil
import subprocess
from typing import List

import cv2
import numpy as np
import pyrealsense2 as rs


def generate_folders(num_cameras: int) -> None:
    """ Create empty folders for the saved images"""
    for n in range(num_cameras):
        path = os.path.join(os.getcwd(), f"camera{n+1}")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def save_frames(frames: List[np.ndarray]) -> None:
    """ Save the camera frames in their respective directories"""
    for n, frame in enumerate(frames):
        save_dir = os.path.join(os.getcwd(), f"camera{n+1}")
        fnum = f"{len(os.listdir(save_dir)):03d}"
        cv2.imwrite(os.path.join(save_dir, f"{fnum}.png"), frame)
    logging.info(f"{fnum} images captured")


def initialise(overwrite_images: bool = True) -> List[rs.pipeline]:
    """ Find and initialises all connected realsense cameras, optionally make new folders for saving captured images"""
    # Initialise cameras
    cameras = []
    for cam in rs.context().devices:
        serial = cam.get_info(rs.camera_info.serial_number)
        logging.info(f"Camera {serial} found, starting stream")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)

        # Try the highest resolution
        try:
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            pipeline.start(config)
        except RuntimeError:
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            pipeline.start(config)

        cameras.append(pipeline)

    # Make output folders
    if overwrite_images:
        generate_folders(len(cameras))

    return cameras


def capture_realsense_sync(cameras) -> None:
    """ Captures simultaneous frames from all connected realsense cameras """

    # Main loop
    while True:
        # Capture frames
        frames = []
        for cam in cameras:
            rs_frames = cam.wait_for_frames()
            rs_col_frame = rs_frames.get_color_frame()
            frames.append(np.asanyarray(rs_col_frame.get_data()))

        # Resize
        frames_resized = []
        for frame in frames:
            scale = 400 / frame.shape[0]
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            frames_resized.append(resized)

        # Display
        cv2.imshow("View", np.hstack(frames_resized))
        k = cv2.waitKey(1)

        # Capture
        if k in [ord('c'), ord("C")]:
            save_frames(frames)

        # Break loop
        elif k in [ord('q'), ord('Q')]:
            break

    # Close
    logging.info("Closing")
    [cam.stop() for cam in cameras]
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    active_cameras = initialise(overwrite_images=False)
    capture_realsense_sync(active_cameras)
    subprocess.call("multical calibrate --boards board.yaml".split(' '))
    subprocess.call("multical vis --workspace_file calibration.pkl".split(' '))
