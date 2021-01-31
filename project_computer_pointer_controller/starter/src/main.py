import cv2
import os
import numpy as np
import logging as log
import time
import datetime
import math

from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder
from argparse import ArgumentParser

# these 2 lines just hide some warning messages.
import warnings
warnings.filterwarnings("ignore")

# create a file to write stats
l_log = open("track_log.txt", "a")
stat_log = open("stats.txt", "a")

# function returns date and time in the format yyyy-mm-dd hh:mm:ss
def timenow():
    time_now = str(datetime.datetime.today()).split()[0] + " " + str(datetime.datetime.today()).split()[1][0:8]
    return time_now

# base code from Project 1 - Deploy a People Counter App at the Edge - main.py file
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file with a trained model.")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Path to a facial landmarks detection model xml file with a trained model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to a head pose estimation model xml file with a trained model.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD are acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+',
                        default=[],
                        help="Example: --flag fd fl hp ge (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame,"
                             "fd for Face Detection Model, fl for Facial Landmark Detection Model"
                             "hp for Head Pose Estimation Model, ge for Gaze Estimation Model.")
    return parser

def get_sine(x):
    return math.sin(x)

def get_cosine(x):
    return math.cos(x)

def get_zero_array():
    return np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)

# Pose estimation from Camera Calibration and 3D Reconstruction
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
def estimate_pose(facial_midpoint, focal_length):
    face_horizontal = int(facial_midpoint[0])
    face_vertical = int(facial_midpoint[1])
    draw_corners = np.zeros((3, 3), dtype='float32')
    draw_corners[0][0] = focal_length
    draw_corners[0][2] = face_horizontal
    draw_corners[1][1] = focal_length
    draw_corners[1][2] = face_vertical
    draw_corners[2][2] = 1
    return draw_corners

def draw_3D_coord_axis(frame, facial_midpoint, heading, pitch, bank, scale, focal_length):
    yaw, pitch, roll = heading, pitch, bank

    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0

    sine_of_pitch = get_sine(pitch)
    cos_of_pitch = get_cosine(pitch)
    sine_heading = get_sine(yaw)
    cos_heading = get_cosine(yaw)
    sine_bank = get_sine(roll)
    cos_bank = get_cosine(roll)

    face_horizontal = int(facial_midpoint[0])
    face_vertical = int(facial_midpoint[1])

    horizontal = np.array([[1, 0, 0],
                          [0, cos_of_pitch, -sine_of_pitch],
                          [0, sine_of_pitch, cos_of_pitch]])
    vertical = np.array([[cos_heading, 0, -sine_heading],
                         [0, 1, 0],
                         [sine_heading, 0, cos_heading]])
    diagonal = np.array([[cos_bank, -sine_bank, 0],
                         [sine_bank, cos_bank, 0],
                         [0, 0, 1]])

    # matrix multiplication - https://docs.python.org/3/whatsnew/3.5.html?highlight=matrix#whatsnew-pep-465
    # using @ for matrix multiplication
    retn = diagonal @ vertical @ horizontal

    draw_corners = estimate_pose(facial_midpoint, focal_length)

    horizontal_axis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    vertical_axis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    diagonal_axis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    diagonal_axis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

    zero_array = get_zero_array()
    zero_array[2] = draw_corners[0][0]

    horizontal_axis = np.dot(retn, horizontal_axis) + zero
    vertical_axis = np.dot(retn, vertical_axis) + zero
    diagonal_axis = np.dot(retn, diagonal_axis) + zero
    diagonal_axis1 = np.dot(retn, diagonal_axis1) + zero

    horizontal_pose_2 = (horizontal_axis[0] / horizontal_axis[2] * draw_corners[0][0]) + face_horizontal
    vertical_pose_2 = (horizontal_axis[1] / horizontal_axis[2] * draw_corners[1][1]) + face_vertical
    posture_2 = (int(horizontal_pose_2), int(vertical_pose_2))
    cv2.line(frame, (face_horizontal, face_vertical), posture_2, (0, 0, 255), 2)

    horizontal_pose_2 = (vertical_axis[0] / vertical_axis[2] * draw_corners[0][0]) + face_horizontal
    vertical_pose_2 = (vertical_axis[1] / vertical_axis[2] * draw_corners[1][1]) + face_vertical
    posture_2 = (int(horizontal_pose_2), int(vertical_pose_2))
    cv2.line(frame, (face_horizontal, face_vertical), posture_2, (0, 255, 0), 2)

    horizontal_pose_1 = (diagonal_axis1[0] / diagonal_axis1[2] * draw_corners[0][0]) + face_horizontal
    vertical_pose_1 = (diagonal_axis1[1] / diagonal_axis1[2] * draw_corners[1][1]) + face_vertical
    posture_1 = (int(horizontal_pose_1), int(vertical_pose_1))

    horizontal_pose_2 = (diagonal_axis[0] / diagonal_axis[2] * draw_corners[0][0]) + face_horizontal
    vertical_pose_2 = (diagonal_axis[1] / diagonal_axis[2] * draw_corners[1][1]) + face_vertical
    posture_2 = (int(horizontal_pose_2), int(vertical_pose_2))

    cv2.line(frame, posture_1, posture_2, (255, 0, 0), 2)
    cv2.circle(frame, posture_2, 3, (255, 0, 0), 2)

    return frame

def infer_on_stream(args):

    input_file_path = args.input
    obj_logger = log.getLogger()
    visual_flag = args.visualization_flag
    if input_file_path == "CAM":
        input_stream = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path):
            obj_logger.error("ERROR: INVALID INPUT PATH")
            exit(1)
        input_stream = InputFeeder("video", input_file_path)

    obj_model_fd = FaceDetection(model_name=args.face_detection_model,
                                 device=args.device,
                                 threshold=args.prob_threshold,
                                 extensions=args.cpu_extension)

    obj_model_fld = FacialLandmarksDetection(model_name=args.facial_landmarks_model,
                                             device=args.device,
                                             extensions=args.cpu_extension)

    obj_model_ge = GazeEstimation(model_name=args.gaze_estimation_model,
                                  device=args.device,
                                  extensions=args.cpu_extension)

    obj_model_hp = HeadPoseEstimation(model_name=args.head_pose_model,
                                      device=args.device,
                                      extensions=args.cpu_extension)

    obj_mouse_controller = MouseController('high', 'fast')  # 'fast')

    sPrec = ""
    sPrec = "test time: " + str(timenow())

    start_time = time.time()
    obj_model_fd.load_model()
    l_log.write("\n Loaded models.....")
    sfd_loadtime = "Face detection: {:.3f} ms".format((time.time() - start_time) * 1000)
    l_log.write("\n" + sfd_loadtime)
    print(sfd_loadtime)

    fl_time = time.time()
    obj_model_fld.load_model()
    sfld_loadtime = "Facial Landmarks detection: {:.3f} ms".format((time.time() - fl_time) * 1000)
    l_log.write("\n" + sfld_loadtime)
    print(sfld_loadtime)

    hpe_time = time.time()
    obj_model_hp.load_model()
    shpe_loadtime = "Head Pose estimation: {:.3f} ms".format((time.time() - hpe_time) * 1000)
    l_log.write("\n" + shpe_loadtime)
    print(shpe_loadtime)

    ge_time = time.time()
    obj_model_ge.load_model()
    sgz_loadtime = "Gaze estimation: {:.3f} ms".format((time.time() - ge_time) * 1000)
    l_log.write("\n" + sgz_loadtime)
    print(sgz_loadtime)
    ttl_time = time.time() - start_time
    print("Total loading time: {:.3f} ms".format(ttl_time * 1000))
    print("---Success loading all models---")
    l_log.write("\n" + "Total loading time: {:.3f} ms".format(ttl_time * 1000))
    l_log.write("\n" + "---Success loading all models---")
    input_stream.load_data()
    print("---Streaming Input video---")
    l_log.write("\n" + "---Streaming Input video--- \n")

    counter = 0
    start_inf_time = time.time()
    for flag, frame in input_stream.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        counter = counter + 1

        try:

            frame_copy = frame.copy()
            face_coordinates, face_image = obj_model_fd.predict(frame_copy)

            if face_coordinates == 0:
                continue

            ret_model_hp = obj_model_hp.predict(face_image)

            left_eye_image, right_eye_image, eye_coord = obj_model_fld.predict(face_image)

            mouse_coordinate, gaze_vector = obj_model_ge.predict(left_eye_image, right_eye_image, ret_model_hp)

            if visual_flag:
                p_frame = frame.copy()
                if 'fd' in visual_flag:
                    if len(visual_flag) != 1:
                        p_frame = face_image
                    else:
                        cv2.rectangle(p_frame, (face_coordinates[0], face_coordinates[1]),
                                      (face_coordinates[2], face_coordinates[3]), (0, 150, 0), 3)
                if 'fl' in visual_flag:
                    if not 'fd' in visual_flag:
                        p_frame = face_image.copy()
                    cv2.rectangle(p_frame, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2], eye_coord[0][3]),
                                  (150, 0, 150))
                    cv2.rectangle(p_frame, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]),
                                  (150, 0, 150))

                # font
                font = cv2.FONT_HERSHEY_DUPLEX
                # org
                org = (20, 20)
                # Blue color in BGR
                color = (0, 0, 0)
                # fontScale
                fontScale = 0.35
                # Line thickness of 1 px
                thickness = 1

                if 'hp' in visual_flag:
                    cv2.putText(p_frame,
                                "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(ret_model_hp[0],
                                                                                 ret_model_hp[1],
                                                                                 ret_model_hp[2]),
                                org, font, fontScale, color, thickness)
                if 'ge' in visual_flag:

                    yaw = ret_model_hp[0]
                    pitch = ret_model_hp[1]
                    roll = out_head_estimate_pose_model[2]
                    focal_length = 950.0
                    scale = 50
                    facial_midpoint = (face_image.shape[1] / 2, face_image.shape[0] / 2, 0)
                    if 'fd' in visual_flag or 'fl' in visual_flag:
                        draw_3D_coord_axis(p_frame, facial_midpoint, yaw, pitch, roll, scale, focal_length)
                    else:
                        draw_3D_coord_axis(frame, facial_midpoint, yaw, pitch, roll, scale, focal_length)

            if visual_flag:
                horizontal_image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(p_frame, (500, 500))))
            else:
                horizontal_image = cv2.resize(frame, (500, 500))

            cv2.imshow('Demo', horizontal_image)
            obj_mouse_controller.move(mouse_coordinate[0], mouse_coordinate[1])

            if key_pressed == 27:
                obj_logger.info("exit key...")
                break

        except:
            print("The video file sent is not supported. Please check the file and resend it.")
            exit()

    rd_time = time.time()
    inference_time = round(rd_time - start_inf_time, 1)
    if inference_time == 0:
        inference_time = 1

    fps = int(counter) / inference_time
    print("Counter {} seconds".format(counter))
    print("Total inference time {} seconds".format(inference_time))
    print("FPS {} frames/second".format(fps))
    print("Total load time {} seconds".format(ttl_time))
    print("End Video ")
    l_log.write("Total inference time " + str(inference_time) + '\n')
    l_log.write("FPS " + str(fps) + '\n')
    l_log.write("Total time " + str(ttl_time) + '\n')

    input_stream.close()
    cv2.destroyAllWindows()

    stat_log.write("\n ---- " + sPrec + " ---- \n")
    stat_log.write("Inference time " + str(inference_time) + '\n')
    stat_log.write("FPS - frames/second " + str(fps) + '\n')
    stat_log.write("Total load time " + str(ttl_time) + '\n')
    stat_log.close()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    tn = ""
    tn = timenow()
    l_log.write("\n" + tn + " Started app \n")
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)

    tn = ""
    tn = timenow()
    l_log.write("\n" + tn + " Completed run \n")
    l_log.close()

if __name__ == '__main__':
    main()
