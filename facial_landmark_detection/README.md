## Facial Landmark Detection Example

Facial landmark detection is the process of detecting facial landmarks like eyes, nose, mouth, etc. in digital images and videos.

This example shows how the SSD MobileNet V2 Face and Face Landmark (SimpleCNN) models may be implemented by using the WebNN API.

### How to Generate SSD MobileNet V2 Face models

SSD MobileNet V2 Face Detection models are trained by Tensorflow Object Detection API with WIDER_FACE dataset. Please go [here](https://github.com/Wenzhao-Xiang/face-detection-ssd-mobilenet) for more training details.

After getting the `frozen_inference_graph.pb`, you can use the following commands by using tensorflow's [`optimize_for_inference`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) tool to convert it to frozen graph.


```
python3 -m tensorflow.python.tools.optimize_for_inference \
--input=${download_model_dir}/frozen_inference_graph.pb \
--output=${out_dir}/frozen_inference_graph_stripped.pb --frozen_graph=True \
--input_names=Preprocessor/sub \
--output_names=\
"BoxPredictor_0/BoxEncodingPredictor/BiasAdd,BoxPredictor_0/ClassPredictor/BiasAdd,\
BoxPredictor_1/BoxEncodingPredictor/BiasAdd,BoxPredictor_1/ClassPredictor/BiasAdd,\
BoxPredictor_2/BoxEncodingPredictor/BiasAdd,BoxPredictor_2/ClassPredictor/BiasAdd,\
BoxPredictor_3/BoxEncodingPredictor/BiasAdd,BoxPredictor_3/ClassPredictor/BiasAdd,\
BoxPredictor_4/BoxEncodingPredictor/BiasAdd,BoxPredictor_4/ClassPredictor/BiasAdd,\
BoxPredictor_5/BoxEncodingPredictor/BiasAdd,BoxPredictor_5/ClassPredictor/BiasAdd" \
--alsologtostderr
```

Use ['TensorFlow Lite converter'](https://www.tensorflow.org/lite/convert) tool to convert frozen graph to tflite model:

```
tflite_convert \
--graph_def_file=${out_dir}/frozen_inference_graph_stripped.pb \
--output_file=${out_dir}/ssd_mobilenetv2_face.tflite \
--input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
--input_shapes=1,300,300,3 --input_arrays=Preprocessor/sub \
--output_arrays=\
"BoxPredictor_0/BoxEncodingPredictor/BiasAdd,BoxPredictor_0/ClassPredictor/BiasAdd,\
BoxPredictor_1/BoxEncodingPredictor/BiasAdd,BoxPredictor_1/ClassPredictor/BiasAdd,\
BoxPredictor_2/BoxEncodingPredictor/BiasAdd,BoxPredictor_2/ClassPredictor/BiasAdd,\
BoxPredictor_3/BoxEncodingPredictor/BiasAdd,BoxPredictor_3/ClassPredictor/BiasAdd,\
BoxPredictor_4/BoxEncodingPredictor/BiasAdd,BoxPredictor_4/ClassPredictor/BiasAdd,\
BoxPredictor_5/BoxEncodingPredictor/BiasAdd,BoxPredictor_5/ClassPredictor/BiasAdd" \
--inference_type=FLOAT --logtostderr
```

Use ['tflite2onnx'](https://github.com/jackwish/tflite2onnx) tool to convert tflite model to onnx model:
```
tflite2onnx ssd_mobilenetv2_face.tflite ssd_mobilenetv2_face.onnx
```

### How to Generate Face Landmark Detection models

Check out [yinguobing/cnn-facial-landmark](https://github.com/yinguobing/cnn-facial-landmark) for more details about this model.

This model is converted from a pre-trained [Simple CNN](https://drive.google.com/file/d/1Nvzu5A9CjP70sDhiRbMzuIwFLnrq2Qpw/view?usp=sharing) model. You can use the following commands to convert your own model.

```
tflite_convert \
--output_file=${out_dir}/face_landmark.tflite \
--graph_def_file=${download_model_dir}/SimpleCNN.pb \
--input_arrays=input_to_float \
--output_arrays=logits/BiasAdd
```

Use ['tflite2onnx'](https://github.com/jackwish/tflite2onnx) tool to convert tflite model to onnx model:
```
tflite2onnx face_landmark.tflite face_landmark.onnx
```

### Usage

### Detect face from images

Choose device, model and layout, in a very few second you will see the predict result for the test image be presented on the page.

You could also click 'Pick Image' button to choose your local image to detect it.

### Detect face from video stream

Here we detect every frame in a live camera, click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt.

Switch model or layout to check variance predict result.