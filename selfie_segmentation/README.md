## WebNN Selfie Segmentation Example

This example demonstrates the [MediaPipe Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation) using the TFLite Web XNNPACK delegate and WebNN delegate, which is built by [tflite-support](https://github.com/huningxin/tflite-support/tree/webnn_delegate_side_module).

### Usage

### Segment images

Choose delegate and model, in a very few second you will see the predict result for the test image be presented on the page. Choose the backgrounds to experience variance portrait segmentation.

You could also click 'Pick Image' button to choose your local image to segment it.

### Segment video stream

Here we segments every frame in a live camera, then combined the results. Click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt, then you will see real-time segmentation results on the page.

Switch delegate or model to check variance predict result.