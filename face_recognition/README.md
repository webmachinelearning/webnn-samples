## Face Recognition Example

Face Recognition is the process of detecting faces of participants by using object detection and checking whether each face was present or not.

This example shows how the SSD MobileNet V2 Face and FaceNet models may be implemented by using the WebNN API.

This example reuses the WebNN implementation of SSD MobileNet V2 Face in [Facial Landmark Detection](../facial_landmark_detection/) example.

```

### How to Generate FaceNet models

Following this [guide](https://medium.com/@estebanuri/converting-sandbergs-facenet-pre-trained-model-to-tensorflow-lite-using-an-unorthodox-way-7ee3a6ed02a3) to convert [Sandberg's FaceNet pre-trained model](https://github.com/davidsandberg/facenet) to TensorFlow Lite.

Use ['tflite2onnx'](https://github.com/onnx/tensorflow-onnx) tool to convert FaceNet TFLite model to ONNX model:

```
python -m tf2onnx.convert --tflite facenet.tflite --output facenet.onnx --inputs-as-nchw input_1 --inputs input_1
```

### Usage

### Recognize face from images

Choose device, model and layout, in a very few second you will see the predict result for the test image be presented on the page.

You could also click 'Pick Faces Image' and 'Pick New Faces Image' buttons to choose your local images to Recognize faces.

### Recognize face from video stream

Here we detect every frame in a live camera, click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt.

Switch model or layout to check variance predict result.

You could also click 'Pick Faces Image' button to choose your local images to recognize faces.