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

class FacialLandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extension = extensions
        # self.threshold=threshold

        try:
            self.model = self._ie_core.read_network(model=self.model_structure, weights=self.model_weights)
        except AttributeError:
            self.model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
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
        self.results = self.net.infer(inputs={self.input_name: self.p_image})
        self.output = self.preprocess_output(self.results, image)
        x_min_le = self.output['x_coord_le'] - 10
        x_max_le = self.output['x_coord_le'] + 10
        y_min_le = self.output['y_coord_le'] - 10
        y_max_le = self.output['y_coord_le'] + 10

        x_min_re = self.output['x_coord_re'] - 10
        x_max_re = self.output['x_coord_re'] + 10
        y_min_re = self.output['y_coord_re'] - 10
        y_max_re = self.output['y_coord_re'] + 10

        self.eye_coord = [[x_min_le, y_min_le, x_max_le, y_max_le],
                          [x_min_re, y_min_re, x_max_re, y_max_re]]
        le_image = image[x_min_le:x_max_le, y_min_le:y_max_le]
        rt_image = image[x_min_re:x_max_re, y_min_re:y_max_re]

        return le_image, rt_image, self.eye_coord

    def check_model(self):
        return

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
        output_blob = {}
        outputs = outputs[self.output_name][0]
        x_coord_le = int(outputs[0] * image.shape[1])
        y_coord_le = int(outputs[1] * image.shape[0])
        x_coord_re = int(outputs[2] * image.shape[1])
        y_coord_re = int(outputs[3] * image.shape[0])

        output_blob['x_coord_le'] = x_coord_le
        output_blob['y_coord_le'] = y_coord_le
        output_blob['x_coord_re'] = x_coord_re
        output_blob['y_coord_re'] = y_coord_re

        return output_blob
