## WebNN Object Detection Example

Object detection is the process of detecting instances of semantic objects of a pre-defined classes in digital images and videos.

This example shows how the Tiny YOLO v2 model from [YAD2K project](https://github.com/allanzelener/YAD2K) and [ONNX model zoo](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz) may be implemented by using the WebNN API. (Note: these two models are all trained on the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.)

### Usage

### Detect images

Choose model and layout, in a very few second you will see the predict result for the test image be presented on the page.

You could also click 'Pick Image' button to choose your local image to detect it.

### Detect video stream

Here we detect every frame in a live camera, click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt.

Switch model or layout to check variance predict result.