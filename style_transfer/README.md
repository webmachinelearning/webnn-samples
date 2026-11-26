## WebNN Fast Style Transfer Example
Fast neural style transfer is the process of taking the style of one image then applying it to the content of another image.
This example shows how the ['Fast Style Transfer' models](https://github.com/acerwebai/VangoghCrazyWorld)
from Vangogh Crazy World project may be implemented by using the WebNN API.

### Usage

### Transfer image style

Select backend, style image of famous paintings on the image gallery, in a very few second you will see the styled image presented on the page.

Use the `Data Type` buttons to switch between `Float32` and `Float16`. Only the data types the selected device reports support for are enabled; switching devices selects the device's preferred data type (`Float16` on NPU, otherwise `Float32`).

You could also click 'Pick Image' button to choose your local image to tranfer to the style you want.

### Transfer video style

Here we transformed every frame in a live camera, then combined the results. Click 'LIVE CAMERA' tab, allow the browser to use your local camera if there's a prompt, then you will see a styled live camera.

Switch the style images to experience various painting styles.

### Note

All image asserts under `images` folder come from https://github.com/acerwebai/VangoghCrazyWorld-Web/tree/master/images, which are licensed under the MIT.