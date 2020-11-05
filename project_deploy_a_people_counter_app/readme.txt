PROJECT - Deploy a People Counter App at the Edge

Steps to download and convert the model to IR.
1. Download faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
2. Unzip the tar file
3. Check the folder.
4. Convert model to IR

The following commands are to be run to download and convert the model.

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

cd faster_rcnn_inception_v2_coco_2018_01_28

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

-----Command to run the application before conversion - we need only one terminal to run pre-conversion------------
---terminal 1---
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

python tf_main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/ -pt 0.4

-----Commands for running the application – we need four terminals to run the application.------
---terminal 1---
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

cd webservice/server/node-server
node ./server.js

---terminal 2---
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
cd webservice/ui
npm run dev

---terminal 3---
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
sudo ffserver -f ./ffmpeg/server.conf

---terminal 4---
python main.py -i /home/workspace/resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4


