## WebNN RNNoise Example
This example shows how the [RNNoise](https://github.com/xiph/rnnoise) baseline implementation of deep learning-based noise suppression model may be implemented using the WebNN API.

### Usage

Choose device preference for inference.

Select one of the existing sample audios from the dropdown list or upload your own one.

Start or pause the original audio playback with the upper media player.

Once the denoising is done, start or pause the denoised audio playback with the lower media player.

### Where did the contents of `process` folder come from?
A web based DSP library using WebAssembly translated from [Rnnoise](https://github.com/xiph/rnnoise), a recurrent neural netork based noise reduction library in C++. We extracted the pre-processing and post-processing parts for compilation using [Emscripten](https://emscripten.org/). Please check out [README.md](process/README.md) in process folder for details.