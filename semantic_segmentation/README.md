## WebNN Semantic Segmentation Example

Semantic segmentation is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category.

This example shows how the [DeepLab V3 MobileNet V2](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz), from TFLite models may be implemented by using the WebNN API. (Note: the DeepLab V3 MobileNet V2 model is trained on the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset.)

## How to Generate Deeplab V3 MobileNet V2 models?

The model referenced in this example only provides frozen graph, we use following tools and commands to convert which to TFLite and ONNX model.

Use tensorflow's [`optimize_for_inference`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) tool to generate a stripped frozen graph:

```
python3 -m tensorflow.python.tools.optimize_for_inference \
--input=./frozen_inference_graph.pb \
--output=./frozen_inference_graph_stripped.pb \
--frozen_graph=True \
--input_names="sub_7" \
--output_names="ArgMax"
```

Use ['TensorFlow Lite converter'](https://www.tensorflow.org/lite/convert) tool to convert stripped frozen graph to TFLite model:

```
tflite_convert \
--graph_def_file=frozen_inference_graph_stripped.pb \
--output_file=deeplab_mobilenetv2.tflite \
--output_format=TFLITE \
--input_format=TENSORFLOW_GRAPHDEF \
--input_arrays=sub_7 \
--output_arrays=ArgMax
```

Use ['tf2onnx'](https://github.com/onnx/tensorflow-onnx) tool to convert TFLite model to ONNX model:

```
python3 -m tf2onnx.convert --opset 13 --tflite deeplab_mobilenetv2.tflite --output deeplab_mobilenetv2.onnx
```

### Usage

### Partition images

Choose device, model and layout, in a very few second you will see the predict result for the test image be presented on the page. Operate the renderer menus to experience variance partitioning effects.

You could also click 'Pick Image' button to choose your local image to partition it.

### Partition video stream

Here we partition every frame in a live camera, then combined the results. Click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt, then you will see real-time partitioning results on the page.

Switch model or layout to check variance predict result.