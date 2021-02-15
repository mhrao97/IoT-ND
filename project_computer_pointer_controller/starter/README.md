# Computer Pointer Controller

Computer Pointer Controller application demonstrates the use of gaze detection model to control the mouse pointer of the computer, from an input video or a live stream of the computer's webcam.

## Project Set Up and Installation
<i> *TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires. </i>   
This project uses Intel's Open Vino enviroment.   
Setup:    
To install Intel Distribution of OpenVINO toolkit, go to https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html     

Install the dependencies - these are listed in the installation instructions. Follow all the instructions till the verification is complete.   
Installation is successful when you see the message   
<tab> Execution successful     
   Demo completed successfully </tab>


The Controller uses the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used identifies the image in the video, the face, the head position and the gaze of the left and right eye. The Computer's mouse pointer follows along the gaze of the person in the video.

This project uses the following pre-trained models from Intel
- Face Detection Model
- Facial Landmarks Detection Model
- Head Pose Estimation Model
- Gaze Estimation Model

Pre-trained models can be installed using the model downloader. I have used the following steps for downloading the models.  
1. Create a temp folder in your C drive - c:\temp; the application and related files will reside in the temp folder.
2. Launch command prompt with Admin rights.
3. Setup the environment variables using commands
   cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
   setupvars.bat    
   (command setupvars.bat initializes openvino environment)
4. Now go to the folder where you want the models downloaded. I have used C:\temp folder for my project. The starter files for this project are within c:\temp folder.    
<img src=c:\temp\starter\temp_folder.png>
I have created a models folder within the starter folder and downloaded the models here.   
<img src=c:\temp\starter\models_folder.png>   

The command used is...   
cd c:\temp\starter\models   

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287/deployment_tools/tools/model_downloader/downloader.py" --name "face-detection-adas-binary-0001"    

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287/deployment_tools/tools/model_downloader/downloader.py" --name "landmarks-regression-retail-0009"    

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287/deployment_tools/tools/model_downloader/downloader.py" --name "head-pose-estimation-adas-0001"    

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287/deployment_tools/tools/model_downloader/downloader.py" --name "gaze-estimation-adas-0002"    

(the above commands will download the following models into c:\temp\starter\models\intel folder)
- Face Detection
- Facial Landmarks Detection
- Head Pose Estimation
- Gaze Estimation Model 

Install pyautogui   
Go to the folder where you want this to be installed (I installed it in C:\Users\M folder)   
Cd c:\users\M 
Pip install pyautogui   
cd c:\temp\starter\src   
 

### Setting up virtual environment ###
1. From command prompt go to the folder where you want the virtual environment to be created. (I created in C:\Users\M folder). Type in the following command.      
<tab> pip install virtualenv </tab>   
<tab> virtualenv venv </tab>    
2. Activate the virtual environment.   
<tab> cd venv\scripts  </tab>   
<tab> activate </tab>   
(venv) prefixed to the C prompt indicates you are now in the virtual environment.   
3. Install dependencies   
<tab> pip install -r c:\temp\starter\requirements.txt   </tab>    
We have already downloaded models at the time of installing OpenVINO.   
4. Initialize OpenVINO environment    
<tab> cd C:\Program Files (x86)\IntelSWTools\openvino\bin\ </tab>   
<tab> setupvars.bat </tab>   
5. We can run our application from C:\temp\starter\src folder.   
<tab> cd C:\temp\starter\src </tab>   



## Demo
<i> *TODO:* Explain how to run a basic demo of your model. </i>   
## Running the application
### on CPU
We are running the application on CPU.  

```
python C:\temp\starter\src\main.py -fd "C:\temp\starter\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -fl "C:\temp\starter\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hp "C:\temp\starter\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -ge "C:\temp\starter\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i C:\temp\CPC\bin\demo.mp4 -d CPU    
```

### on GPU 

```
python C:\temp\starter\src\main.py -fd "C:\temp\starter\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -fl "C:\temp\starter\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hp "C:\temp\starter\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -ge "C:\temp\starter\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i C:\temp\CPC\bin\demo.mp4 -d GPU
```

### on FPGA 

```
python C:\temp\starter\src\main.py -fd "C:\temp\starter\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -fl "C:\temp\starter\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hp "C:\temp\starter\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -ge "C:\temp\starter\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i C:\temp\CPC\bin\demo.mp4 -d HETERO:FPGA,CPU
```

### on NSC2

```
python C:\temp\starter\src\main.py -fd "C:\temp\starter\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -fl "C:\temp\starter\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hp "C:\temp\starter\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -ge "C:\temp\starter\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i C:\temp\CPC\bin\demo.mp4 -d MYRIAD
```

## Documentation
<i> *TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.</i>   
The code for this project is split into various python files.
1. main.py   
<tab> main file to run the application </tab>
2. face_detection.py   
<tab> This file uses face_detection model to detect the face from the input video. Pre-processes the input video frame, performs inference on it and detects the face. </tab>   
3. facial_landmarks_detection.py   
<tab> This file uses landmarks_regression_retail model. Takes in the detected face from face_detection.py as input, pre-processes the face, performs inference and detects the eye landmarks. </tab>   
4. head_pose_estimation.py   
<tab> This file uses head_pose_estimation model. Takes the detected face as input, pre-processes the face, performs inference and detects the position of the head by predicting yaw - roll - pitch angles.   </tab>   
5. gaze_estimation.py   
<tab> This file uses gaze_estimation_adas model. Takes the left eye, right eye, head pose angles as inputs, pre-processes, perform inference and predicts the gaze vector. </tab>   
6. input_feeder.py   
<tab> Contains InputFeeder class which initialize VideoCapture as per the user argument and returns the frames.    </tab>
7. mouse_controller.py   
<tab> Contains MouseController class which takes x, y coordinate value, speed, precisions and based on these values moves the mouse pointer using pyautogui library. </tab>   
The directory structure used is as per the recommendations for the project:    starter folder contains bin, models and src sub-folders   
  bin folder contains the input video file   
  models folder contains the pre-trained models used in the application   
  src folder contains the source code (all py files) to run the application


The application takes the following arguements:   
1. -fd	: (required),Path to a face detection model xml file with a trained model.   
2. -fl	: (required),Path to a facial landmarks detection model xml file with a trained model.   
3. -hp	: (required),Path to a head pose estimation model xml file with a trained model.   
4. -ge	: (required),Path to a gaze estimation model xml file with a trained model.   
5. -i	: (required),Path to image or video file or CAM   
6. -l	: (optional),MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.   
7. -d	: Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified (CPU by default)   
8. -pt	: Probability threshold for detections filtering (0.5 by default)   
9. -flag	: (optional),Example: fd fl hp ge (Seperate each flag by space) to see the visualization of different model outputs of each frame, fd for Face Detection Model, fl for Facial Landmark Detection Model, hp for Head Pose Estimation Model, ge for Gaze Estimation Model.   


## Benchmarks
<i> *TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc. </i>   
I ran the application on my local PC with the following configuration:   
Intel Core i5-8265U 1.6GHz with Turbo Boost upto 3.9GHz

## Results
<i> *TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models. </i>   
### Following is the Model Size for each of the models used in the application

* <b>face-detection-adas-binary-0001</b>

| Type         | Model Size |
|--------------|---------------|
|  FP32-INT1   |  1.86M        |

* <b>head-pose-estimation-adas-0001</b>

| Type         | Model Size |
|--------------|---------------|
|  FP16   |  3.69M       |
|  FP16-INT8   |  2.05M        |
|  FP32   |  7.34M        |

* <b>landmarks-regression-retail-0009</b>

| Type         | Model Size |
|--------------|---------------|
|  FP16   |  413KB      |
|  FP16-INT8   | 314KB        |
|  FP32   |  786KB       |

* <b>gaze-estimation-adas-0002</b>

| Type         | Model Size |
|--------------|---------------|
|  FP16   |  3.65M        |
|  FP16-INT8   |  2.05M      |
|  FP32   |   7.24M       |


## Model Performance

| Activity / Precision | 	FP16 |  	FP16-INT8 | 	FP32 | 
| ------------ | ----- | ----- | ---- | ----- |
| Face detection | 	187.422 ms | 	204.493 ms | 	205.180 ms | 
| Facial landmarks detection | 	156.212 ms | 	217.537 ms | 	164.585 ms | 
| Head pose estimation | 	125.004 ms | 	239.436 ms | 	126.592 ms | 
| Gaze estimation | 	140.590 ms | 	302.748 ms | 	161.144 ms | 
| Total loading time | 	609.228 ms | 	964.214 ms | 	657.502 ms | 
| Inference time | 	97.1 | 	96.8 | 	97.8 | 
| Frames per second | 	0.607621009 | 	0.609504132 | 	0.603271984 | 
| Total loading time | 	0.609228134 | 	0.964214325 | 	0.657501936 | 
    
    
I ran the application on my local machine in virtual environment. The results are posted above. In referencing MouseController, I have the precision set to high and mouse speed set to fast.   
When I ran the application for all three precisions FP16, FP16-INT8 and FP32, there was marginal difference in inference time and frames per second. Total load time for the models varied a bit with FP16-INT taking longest.   


<img src=./screen_shots/FP16.png>

<img src=./screen_shots/FP16-INT8.png>

<img src=./screen_shots/FP32.png>


## Stand Out Suggestions
<i>This is where you can provide information about the stand out suggestions that you have attempted.</i>   
I compared the model performances on inference time, frames per second and the loading time of the model.

Models were run with different precisions. Precision affects the accuracy and speed. If the precision is lowered, it lowers the Model size and the inference is faster. This may affect accuracy as some of the important information may be lost. Selecting a Precision is a compromise between accuracy and inference time.    

Initially, I ran the application with precision FP16 keeping the speed of the mouse at 'fast' while calling the MouseController. When I ran the application with precision FP16-INT8, the mouse pointer lost track and went off the computer screen. The application got aborted with an error message with 'fail safe check'. When I re-ran the application, I got the same error. So, I changed the mouse speed to medium and ran the application for all three precisions again. Although the application ran for all three models without any errors, inference was slow.   

In my next test, I set the precision to high and mouse speed to fast while calling the MouseController. With this change, the application ran faster and the mouse movement was within a certain frame.   In this application, there is marginal difference in inference and frames per second for each of the models, with different precisions. Total load time for FP16-INT8 was higher.    


### Edge Cases
<i> There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.</i>   

Lighting helps in identifying an image better, in that, if the video being covered is bright, the face and the gaze of the person can be identified with better accuracy. If the area being covered is dark, detecting the face could be difficult - this may result in sending the mouse pointer off the track.   

Camera focal length/image size helps in identifying the face better. If the image captured is of high resolution, the accuracy is improved.   

I ran the application with mouse speed set to fast. With faster mouse speed, application got aborted for lower precision. I was able to run the application for all three precision with mouse speed set to medium. When I set the MouseController precision to high and mouse speed to fast, the results were better. There is scope for improvement here. Perhaps an area around the video frame can be defined and letting the mouse move only within the defined space. With this feature, the precision and speed of the MouseController can be set to user's choise and we may be able to run the application smoothly.    
