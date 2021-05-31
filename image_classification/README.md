## WebNN Image Classfication Example

Image classfication is the process of taking images as input and classifying the major object in the image into a set of pre-defined classes.

This example shows how the [MobileNet V2](https://github.com/onnx/models/tree/master/vision/classification/mobilenet), [SqueezeNet 1.1](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) from ONNX models and [MobileNet V2](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) [SqueezeNet 1.0](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz) from TFLite models may be implemented by using the WebNN API.

### Usage

### Classify images

Choose model and layout, in a very few second you will see the predict result for the test image be presented on the page.

You could also click 'Pick Image' button to choose your local image to classfy it.

### Classify video stream

Here we classfy every frame in a live camera, then combined the results. Click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt, then you will see real-time classfication results on the page.

Switch model or layout to check variance predict result.