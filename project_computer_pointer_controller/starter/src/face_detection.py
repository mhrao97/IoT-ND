'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore

# these 2 lines just hide some warning messages.
import warnings
warnings.filterwarnings("ignore")

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.extension = extensions
        self.facial_image = None
        self.co_ords = None
        self.input_dict = None
        self.p_image = None
        self.net = None

        try:
            self.model =  self._ie_core.read_network(model=self.model_structure, weights=self.model_weights)
        except AttributeError:
            self.model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.infer_request = None
        '''

        return

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name = self.device, num_requests=1)

        ### Check for supported layers ###
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        layers = self.model.layers.keys()
        unsupported_layers = False
        for l in layers:
            if l not in supported_layers:
                unsupported_layers = True

        ### Add any necessary extensions ###
        if unsupported_layers == True:
            self.core.add_extension(self.extension, self.device)

        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.p_image = self.preprocess_input(image)
        self.input_dict = self.net.infer({self.input_name: self.p_image})
        self.detect_faces = self.preprocess_output(self.input_dict, image)
        if len(self.detect_faces) == 0:
            log.error("The application did not detect any faces in the image. Will continue processing next frame")
            return 0, 0

        self.co_ords = self.detect_faces[0]
        facial_image = image[self.co_ords[1]:self.co_ords[3],
                             self.co_ords[0]:self.co_ords[2]]

        return self.detect_faces, facial_image

    def check_model(self):
        return

    def draw_outputs(self, co_ords, image):
        width, height = image.shape[1], image.shape[0]
        people_frame = []
        for box in co_ords[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                # image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # people_frame.append(box)
                people_frame.append([xmin, ymin, xmax, ymax])
        return people_frame

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_img = np.copy(image)
        no_img, no_ch, height, width = self.input_shape
        input_img = cv2.resize(input_img, (width,height), interpolation = cv2.INTER_AREA)
        input_img = input_img.transpose((2,0,1))
        input_img = input_img.reshape(1, *input_img.shape)
        return input_img

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        '''        
        output_blob = {}
        for out in outputs:
            output_name = self.output_name
            output_image = out
            output_blob[output_name] = output_image
        return output_blob
        '''
        width, height = image.shape[1], image.shape[0]
        people_frame = []
        for box in outputs[self.output_name][0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                # image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # people_frame.append(box)
                people_frame.append([xmin, ymin, xmax, ymax])
        return people_frame
