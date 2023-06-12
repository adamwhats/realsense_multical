import json

import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == '__main__':
    # Load poses
    with open('calibration.json', 'r') as fp:
        camera_poses = json.load(fp)['camera_poses']

    # Parse poses and change from ROS2 to optical coordinates convention
    for pose in camera_poses:
        # Ignore base camera
        if pose.find('_to_') == -1:
            continue

        # Find tf matrix between optical frames outputted by calibration
        t_opt = np.asarray(camera_poses[pose]['T'])
        r_opt = np.asarray(camera_poses[pose]['R'])
        tf_cam2_opt_to_cam1_opt = np.vstack([np.hstack([r_opt, t_opt.reshape(3, 1)]), [0, 0, 0, 1]])

        # Find the tf between camera links (ROS coordinate convention)
        tf_cam1_opt_to_link = np.array([[0.002, -1.000,  0.005, -0.059],
                                        [0.003, -0.005, -1.000, -0.000],
                                        [1.000,  0.002,  0.003,  0.000],
                                        [0.000,  0.000,  0.000,  1.000]])

        tf_cam2_opt_to_link = np.array([[0.005, -1.000, -0.002,  0.015],
                                        [0.000,  0.002, -1.000, -0.000],
                                        [1.000,  0.005,  0.000, -0.000],
                                        [0.000,  0.000,  0.000,  1.000]])

        tf_mat_ros = np.linalg.inv(tf_cam2_opt_to_link) @ tf_cam2_opt_to_cam1_opt @ tf_cam1_opt_to_link

        # Convert to (X, Y, Z, RX, RY, RZ, RW)
        r = Rotation.from_matrix(np.asarray(tf_mat_ros[:3, :3])).as_quat()
        t = tf_mat_ros[:3, 3]
        tf_string = [str(round(n, 3)) for n in list(np.hstack([t, r]))]
        print(f"{pose}, {tf_string}")
        print(
            f"ros2 run tf2_ros static_transform_publisher --x {tf_string[0]} --y {tf_string[1]} --z {tf_string[2]} --qx {tf_string[3]} --qy {tf_string[4]} --qz {tf_string[5]} --qw {tf_string[6]} --frame-id camera2_link --child-frame-id camera1_link")
