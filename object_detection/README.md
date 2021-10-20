## WebNN Object Detection Example

Object detection is the process of detecting instances of semantic objects of a pre-defined classes in digital images and videos.

This example shows how the Tiny YOLO v2 model from [YAD2K project](https://github.com/allanzelener/YAD2K) and [ONNX model zoo](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz), [SSD MobileNet V1 models](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) may be implemented by using the WebNN API. (Note: Tiny YOLO V2 models are trained on the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset, SSD MobileNet V1 models are trained on the [COCO](https://cocodataset.org/#home) dataset.)


### How to Generate SSD MobileNet V1 models

Since the original [SSD MobileNet V1 model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) contains customized preprocess and postprocess graphs, we only implement a cut model with WebNN API, you can generate the cut model via following commands:

Here is the converter command for removing the preprocess and postprocess graphs by using tensorflow's [`optimize_for_inference`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) tool:

```
python3 -m tensorflow.python.tools.optimize_for_inference \
--input=./frozen_inference_graph.pb \
--output=./frozen_inference_graph_stripped.pb --frozen_graph=True \
--input_names=Preprocessor/sub \
--output_names='concat,concat_1' \
--alsologtostderr
```

Use ['TensorFlow Lite converter'](https://www.tensorflow.org/lite/convert) tool to convert frozen graph to tflite model:

```
tflite_convert \
--graph_def_file=./frozen_inference_graph_stripped.pb \
--output_file=./ssd_mobilenet_v1_coco.tflite \
--input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
--input_shapes=1,300,300,3 --input_arrays=Preprocessor/sub \
--enable_v1_converter \
--output_arrays='concat,concat_1' \
--inference_type=FLOAT --logtostderr
```

Use ['tflite2onnx'](https://github.com/jackwish/tflite2onnx) tool to convert tflite model to onnx model:
```
tflite2onnx ssd_mobilenet_v1_coco.tflite ssd_mobilenet_v1_coco.onnx
```

### Usage

### Detect images

Choose device, model and layout, in a very few second you will see the predict result for the test image be presented on the page.

You could also click 'Pick Image' button to choose your local image to detect it.

### Detect video stream

Here we detect every frame in a live camera, click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt.

Switch model or layout to check variance predict result.