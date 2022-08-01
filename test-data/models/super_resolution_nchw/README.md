`.npy` files under this folder are downloaded from model of [sub_pixel_cnn_2016](https://github.com/onnx/models/blob/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz),
which is licensed under the Apache 2.0.

### How to Generate Super Resolution (sub_pixel_cnn_2016) Model
Download and extract [Super Resolution (with sample test data)](https://github.com/onnx/models/blob/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz), then use https://netron.app/ to open `.ONNX`, Navigate to each node "layer" per W (=Weight) and B (=Biases), click "+", then click save.

Convert to `.npy` formatted files by:
```sh
python3 gen_nhwc_test_data.py -d path\to\super_resolution -n super_resolution
```