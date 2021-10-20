## WebNN Image Classfication Example

Image classfication is the process of taking images as input and classifying the major object in the image into a set of pre-defined classes.

This example shows how the [MobileNet V2 ONNX](https://github.com/onnx/models/tree/master/vision/classification/mobilenet), [SqueezeNet 1.1 ONNX](https://github.com/onnx/models/tree/master/vision/classification/squeezenet), [ResNet50 V2 ONNX](https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v2-7.tar.gz) from ONNX models and [MobileNet V2 TFLite](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) [SqueezeNet 1.0 TFLite](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz), [ResNet101 V2 TFLite](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz), [ResNet50 V2 TFLite](https://storage.googleapis.com/download.tensorflow.org/models/tflite/resnet_v2_50_2018_03_27.zip) from TFLite models may be implemented by using the WebNN API.

### Usage

### Classify images

Choose device, model and layout, in a very few second you will see the predict result for the test image be presented on the page.

You could also click 'Pick Image' button to choose your local image to classfy it.

### Classify video stream

Here we classfy every frame in a live camera, then combined the results. Click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt, then you will see real-time classfication results on the page.

Switch model or layout to check variance predict result.