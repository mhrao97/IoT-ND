"""People Counter."""
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


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

import numpy as np

from argparse import ArgumentParser
from inference import Network

from random import randint
import datetime
# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# create a log file to debug
l_log = open("track_log.txt", "a")

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
                        help="Path to an xml file with a trained model.")
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
    return parser

def draw_boxes(frame, result, width, height, prob_threshold):
    people_count = 0
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    
    '''     Draw bounding boxes onto the frame.'''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            people_count += 1
            pr_time = ""
            pr_time = str(timenow())
            wr_draw = ""
            wr_draw = pr_time + " fm dboxes people_count xmin ymin xmax ymax "
            wr_draw = wr_draw + str(people_count) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
            l_log.write("\n" + wr_draw)
    return frame, people_count, xmin, ymin, xmax, ymax

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None
    client = mqtt.Client()
    client.disconnect()     # disconnect if already connected.
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Create a flag for single images
    image_flag = False
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    model = args.model
    device = args.device
    cpu_ext = args.cpu_extension

    infer_network.load_model(model=model, device=device, cpu_extension=cpu_ext)
    net_input_shape = infer_network.get_input_shape()
    input_shape = net_input_shape['image_tensor']

    ### TODO: Handle the input stream ###
    input_stream = args.input
    if args.input.upper() == "CAM":
        cv2.VideoCapture(0)
    elif args.input.endswith(".jpg") or args.input.endswith(".bmp") or args.input.endswith(".png"):
        image_flag = True
    else:
        assert os.path.isfile(args.input), "System is unable to recognize your input. Please provide a valid input"

        
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('vino_out.mp4', 0x00000021, 30, (width,height))

    # variables for duration
    start_time = 0
    end_time = 0
    duration = 0
    total_duration = 0
    avg_duration = 0

    current_count = 0
    total_count = 0
    last_count = 0
    prev_current_count = 0
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    prev_x_min = 0
    prev_y_min = 0
    prev_x_max = 0
    prev_y_max = 0
    
    # text 
    text = 'Person '
    text_2 = 'Total  ' 
    # font
    #font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_DUPLEX 
    # org 
    org = (80, 320)
    org_2 = (80, 380)
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 0, 0)
    # Line thickness of 2 px 
    thickness = 2
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        """  
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        """
        p_frame = preprocess_image(frame, input_shape[3], input_shape[2])
        
        ### TODO: Start asynchronous inference for specified request ###
        # model sent is inception, it needs the input as tensor.
        p_input = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
        infer_network.exec_net(p_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            # frame = cv2.putText(frame, "Prev count " + str(prev_current_count), (20, 190), font, 1,  color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, "Curr count " + str(current_count), (20, 230), font, 1,  color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, "Total count " + str(total_count), (20, 270), font, 1,  color, thickness, cv2.LINE_AA, False)
            pr_time = ""
            pr_time = str(timenow())
            frame = cv2.putText(frame, "Date Time " + pr_time, (20, 310), font, 1,  color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, "Duration " + str(duration), (20, 350), font, 1,  color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, "Avg Duration " + str(avg_duration), (20, 390), font, 1,  color, thickness, cv2.LINE_AA, False)

            
            ### TODO: Extract any desired stats from the results ###
            out_frame, current_count, x_min, y_min, x_max, y_max = draw_boxes(frame, result, width, height, prob_threshold)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            wr_fr_draw = ""
            pr_time = ""
            pr_time = str(timenow())
            wr_fr_draw = pr_time + " returned current_count x_min y_min x_max y_max " + str(current_count) + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max)
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
            
            """
            client.publish("person", json.dumps({"count": current_count}))
            client.publish("total ", json.dumps({"total": total_count}))
            client.publish("duration ", json.dumps({"duration": duration}))
            client.publish("person/duration ", json.dumps({"person/duration": avg_duration}))
            """
            client.publish("person", json.dumps({"count": current_count}))
            client.publish("person ", json.dumps({"total": total_count}))
            client.publish("duration ", json.dumps({"duration": duration}))
            client.publish("person/duration ", json.dumps({"duration": avg_duration}))

            publish_cnt = ""
            pr_time = ""
            pr_time = timenow()
            publish_cnt = pr_time + " published to M server count " + str(current_count)
            l_log.write("\n" + publish_cnt)
            
            last_count = current_count
        prev_current_count = current_count
        
        pr_time = ""
        pr_time = str(timenow())
        l_log.write("\n" + pr_time + " prev_current_count = current_count. " + str(prev_current_count))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        out.write(out_frame)

        ### TODO: Write an output image if `single_image_mode` ###
        if key_pressed == 27:
            break
            
    client.publish("person ", json.dumps({"count": total_count}))
    client.publish("person/duration ", json.dumps({"duration": avg_duration}))

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


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
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    
    tn = ""
    tn = timenow()
    l_log.write("\n" + tn + " Completed run \n")
    l_log.close()


if __name__ == '__main__':
    main()
