"""People Counter in tensorflow."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
"""
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
"""

import os
import sys
import time
import socket
import json
import cv2

import numpy as np
import tensorflow as tf

import logging as log

from argparse import ArgumentParser

import datetime

# create a log file to debug
l_log = open("tf_track.txt", "a")

from sys import platform

# function returns date and time in the format yyyy-mm-dd hh:mm:ss
def timenow():
    time_now = str(datetime.datetime.today()).split()[0] + " " + str(datetime.datetime.today()).split()[1][0:8]
    return time_now

# function to pre-process image
def preprocess_image(input_image, width, height):
    
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, *image.shape)
    
    return image

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an saved_model.pb file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-framerate", type=int, default=24,
                       help="frame rate - 1 by default")
    parser.add_argument("-streammode", type=str, default="video",
                        help="video or image - video by default")
    return parser

def infer_on_stream(args, client=None):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    image_flag = False
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        image_flag = True

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    model_dir = args.model

    # Read the graph.
    with tf.gfile.FastGFile(model_dir + 'frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    ### Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('tf_out.mp4', 0x00000021, 30, (width,height))
    
    # variables for duration
    start_time = 0
    end_time = 0
    duration = 0
    total_duration = 0
    avg_duration = 0    
    
    # Define variables for counts
    current_count = 0
    total_count = 0
    frame_count = 0
    prev_current_count = 0
    frame_rate = args.framerate
    
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    prev_x_min = 0
    prev_y_min = 0
    prev_x_max = 0
    prev_y_max = 0
    
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    
        ### Loop until stream is over ###
        while cap.isOpened():
            # increase the frame count
            frame_count += 1

            ### Read from the video capture ###
            flag, frame = cap.read()
            if not flag:
                break
            key_pressed = cv2.waitKey(60)

            ### Pre-process the image as needed ###     
            img = frame 
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            
            # Run the model
            output = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            # font
            #font = cv2.FONT_HERSHEY_SIMPLEX
            font = cv2.FONT_HERSHEY_DUPLEX 
            # fontScale 
            fontScale = 1
            # Blue color in BGR 
            color = (255, 0, 0)
            # Line thickness of 2 px 
            thickness = 2

            frame = cv2.putText(frame, "Pre-conversion ", (20, 30), font, 1,  color, thickness, cv2.LINE_AA, False)
            pr_time = ""
            pr_time = str(timenow())
            frame = cv2.putText(frame, "Date Time " + pr_time, (20, 390), font, 1,  color, thickness, cv2.LINE_AA, False)

            # Visualize detected bounding boxes.            
            num_detections = int(output[0][0])
            for i in range(num_detections):
                classId = int(output[3][0][i])
                score = float(output[1][0][i])
                bbox = [float(v) for v in output[2][0][i]]
                if score > prob_threshold and classId == 1:
                    current_count += 1
                    x_min = int(bbox[1] * cols)
                    y_min = int(bbox[0] * rows)
                    x_max = int(bbox[3] * cols)
                    y_max = int(bbox[2] * rows)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
                            
            wr_fr_draw = ""
            pr_time = ""
            pr_time = str(timenow())
            wr_fr_draw = pr_time + " box drawn "
            l_log.write("\n" + wr_fr_draw)

            # store previous co-ordinates
            if x_min > 0 and y_min > 0:
                prev_x_min = x_min
                prev_y_min = y_min
                prev_x_max = x_max
                prev_y_max = y_max
                
            # if previous co-ordinates are > 0 and current co-ordinates are all 0's then the person has exited the frame. Get the end time to calculate duration.
            if (prev_x_min >= 600 and prev_y_min > 0 and prev_x_max > 0) and (x_min == 0 and y_min == 0 and x_max == 0 and y_max == 0):
                end_time = int(time.time())
                pr_time = str(timenow())
                wr_fr_draw = pr_time + " previous x_min y_min x_max y_max " + str(prev_x_min) + " " + str(prev_y_min) + " " + str(prev_x_max) + " " + str(prev_y_max)
                l_log.write("\n" + wr_fr_draw)
                # reset previous co-ordinates
                prev_x_min = x_min
                prev_y_min = y_min
                prev_x_max = x_max
                prev_y_max = y_max
                if (end_time > start_time) and total_count >= 1:
                    duration = int(end_time - start_time)
                    total_duration += duration
                    avg_duration = total_duration / total_count
                    
                pr_time = str(timenow())
                wr_fr_draw = pr_time + " exit-frame current_count duration ttl_dur avg_dur x_min y_min x_max y_max " + str(current_count) + " " + str(duration) + " " + str(total_duration) + " " + str(avg_duration) + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max)
                l_log.write("\n" + wr_fr_draw)

            if prev_current_count == 0 and current_count == 1 and x_min < 330:
                start_time = int(time.time())
                total_count = total_count + current_count
                wr_count = ""
                pr_time = ""
                pr_time = timenow()
                wr_count = pr_time + " x_min prev_current_count current_count total_count  "
                wr_count = wr_count + str(x_min) + " " + str(prev_current_count) + " " + str(current_count) + " " + str(total_count)
                l_log.write("\n" + wr_count)
            
            ### Send the frame ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
            if image_flag:
                cv2.imwrite('output_image.jpg', frame)
            else:
                out.write(frame)
        
        prev_current_count = current_count
        pr_time = ""
        pr_time = str(timenow())
        l_log.write("\n" + pr_time + " prev_current_count = current_count. " + str(prev_current_count))

        out.write(frame)    
    
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    out.release()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    tn = ""
    tn = timenow()
    l_log.write("\n" + tn + " Started app \n")
    st_time = time.time()
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)

    tn = ""
    tn = timenow()
    l_log.write("\n" + tn + " Completed run \n")
    dur = int(time.time() - st_time)
    l_log.write("Total run time " + str(dur))
    l_log.close()

if __name__ == '__main__':
    main()

