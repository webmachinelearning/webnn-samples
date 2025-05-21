## WebNN Selfie Segmentation Example

Selfie Segmentation is the task of segmenting the prominent humans in the scene.

This example demonstrates how the [MediaPipe Selfie Segmentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md) TFLite models may be implemented by using the WebNN API.

### Usage

### Segment images

Choose backend, data type and resolution type, in a very few second you will see the predict result for the test image be presented on the page. Choose the backgrounds to experience variance portrait segmentation.

You could also click 'Pick Image' button to choose your local image to segment it.

### Segment video stream

Here we segments every frame in a live camera, then combined the results. Click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt, then you will see real-time segmentation results on the page.

Switch backend, data type or resolution type to check variance predict result.