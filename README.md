[![lint](https://github.com/webmachinelearning/webnn-samples/workflows/lint/badge.svg)](https://github.com/webmachinelearning/webnn-samples/actions)
[![deploy](https://github.com/webmachinelearning/webnn-samples/workflows/deploy/badge.svg)](https://github.com/webmachinelearning/webnn-samples/actions)

# WebNN API Samples
This repository contains a collection of samples and examples demonstrating Web Neural Network API (WebNN) usage in web applications. Web Neural Network API (WebNN) is a JavaScript API that provides a high-level interface for performing machine learning computations on neural networks in web applications. With WebNN, developers can leverage hardware acceleration to efficiently run inference tasks on various devices, including CPUs, GPUs, and dedicated AI accelerators. It simplifies the integration of machine learning models into web apps, opening up new possibilities for interactive experiences and intelligent applications right in the browser.

## Repository Structure
This repository hosts a wide range of samples and examples that showcase different use cases and functionalities of WebNN. Here's an overview of the directory structure:    

* [Code Editor](/code): This is a Code Editor used for evaluating, reviewing and modifying WebNN sample codes interactively in web browser.
* [Face recognition](/face_recognition): This directory contains examples of SSD MobileNet V2 Face and Face Landmark (SimpleCNN) model implementation.
* [Facial landmark detection](/facial_landmark_detection): This directory contains examples of SSD MobileNet V2 Face and Face Landmark (SimpleCNN) model implementation.
* [Image classification](/image_classification): This directory contains examples demonstrating image classification using pre-trained models with WebNN.
* [LeNet](/lenet): This example showcases the LeNet-based handwritten digits classification by WebNN API.
* [NNotepad](/nnotepad): This is a browser-based playground for experimenting with WebNN expressions without boilerplate code.
* [NSNet2](/nsnet2): This example shows how to implement the NSNet2 baseline implementation of a deep learning-based noise suppression model.
* [Object detection](/object_detection): Samples showcasing object detection tasks using WebNN with pre-trained models.
* [RNNoise](/rnnoise): This example shows the RNNoise baseline implementation of a deep learning-based noise suppression model.
* [Selfie segmentation](/selfie_segmentation): This example demonstrates the MediaPipe Selfie Segmentation using the TFLite Web XNNPACK delegate and WebNN delegate, built by tflite-support.
* [Semantic segmentation](/semantic_segmentation): This directory contains examples of implementing the DeepLab V3 MobileNet V2, from TFLite models.
* [Style transfer](/style_transfer): Explore examples highlighting the artistic possibilities of WebNN by applying style-transfer techniques to images.

## Requirements
You will require a compatible browser that supports Web Neural Network API (WebNN) to run the samples in this repository. Currently, Chrome and Edge browsers provide support for WebNN.

## Getting Started
To get started, follow these steps:    
1. Clone the repository to your local machine and navigate to it:
 ```bash
> git clone --recurse-submodules https://github.com/webmachinelearning/webnn-samples
> cd webnn-samples
> npm install
> npm run start
```
2. Navigate to the desired sample directory that you want to explore.
3. Read the accompanying README.md file for the sample to understand its purpose, requirements, and implementation details.
4. Follow the instructions provided in the README to set up the necessary dependencies and run the sample.
5. Keep in mind that WebNN currently supports the CPU backend only on Chrome or Edge browsers, and it requires enabling the experimental web platform features flag (see below). Ensure you have this flag enabled in your browser to fully experience WebNN functionality.
6. Experiment with the code and explore how WebNN can enhance machine learning tasks in the browser, navigating to http://localhost:8080.

### WebNN Installation Guides

To get started with WebNN on Intel AI PCs you will need:
* Window 11, version 21H2 or newer
* It's recommended to install the latest [Intel® Arc™ & Iris® Xe Graphics](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html) on Windows for improved WebNN compatibility and performance

1. Download and install the latest [Chrome Canary](https://www.google.com/chrome/canary/) or [Edge Canary](https://www.microsoft.com/en-us/edge/download/insider?form=MA13FJ)
2. Navigate to `about://flags` in browser address bar
3. Search for `Enables WebNN API` and change it to "Enabled"
4. Exit browser

#### Running WebNN on CPU or GPU
1. Launch Chrome Canary or Edge Canary

#### Running WebNN on NPU
At present, the [image classification](https://webmachinelearning.github.io/webnn-samples/image_classification/) and [object detection](https://webmachinelearning.github.io/webnn-samples/object_detection/) samples support NPU.

* Window 11, version 24H2 or newer
* It's recommended to install the latest [Intel® Core™ Ultra NPU Driver on Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) for improved WebNN compatibility and performance
* **Google Chrome Canary:**
  1. Download the latest redistributable [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.2), rename the "microsoft.ai.directml.\<version>.nupkg" to "microsoft.ai.directml.\<version>.nupkg.zip" and extract it
  2. Copy "\bin\x64-win\DirectML.dll" to "%LOCALAPPDATA%\Google\Chrome SxS\Application\\<version>\"
      - "%LOCALAPPDATA%" means "C:\Users\<username>\AppData\Local\"
      - Note that Chrome Canary frequently updates automatically. When this occurs, you'll need to recopy the DirectML.dll to the new version's directory
  3. Launch Chrome Canary in Windows Command Line:
  ```bash
  "%LOCALAPPDATA%\Google\Chrome SxS\Application\chrome.exe" --use-redist-dml --disable_webnn_for_npu=0
  ```

* **Microsoft Edge Canary:**
  1. Ensure the DirectML.dll was downloaded automatically (may take several minutes):
      - Launch Edge Canary
      - Go to "%LOCALAPPDATA%\Microsoft\Edge SxS\User Data", check the "EdgeOnnxRuntimeDirectML\<version>\DirectML.dll" exists
      - "%LOCALAPPDATA%" means "C:\Users\<username>\AppData\Local\"
      - Exit Edge Canary
  2. Launch Edge Canary in Windows Command Line:
  ```bash
  "%LOCALAPPDATA%\Microsoft\Edge SxS\Application\msedge.exe" --disable_webnn_for_npu=0
  ```

* Notes:
1. There is an intermittent issue with the Intel NPU driver that causes failure of NPU adapter creation. The `WebNN(NPU)` backend button in the samples will be disabled with message "Unable to find a capable adapter". If you encounter this issue, please relaunch your browser and try again.
2. The flag `disable_webnn_for_npu` is set to true by default to disable WebNN on NPU due to the aforementioned issue. To bypass this, use `--disable_webnn_for_npu=0`. This flag will be removed once the issue is resolved.
3. Running WebNN on NPU requires a higher version of DirectML.dll than the one in the Windows system. Using the `--use-redist-dml` flag will allow Google Chrome Canary to load the downloaded DirectML.dll with a sufficiently high version.

## Support and Feedback
If you encounter any issues or have feedback on the WebNN Samples, please open an issue on the repository. We appreciate your input and will strive to address any problems as quickly as possible.

You can also join our [community forum](https://webmachinelearning.github.io/) for general questions and discussions about WebNN.

## Contributing
We welcome contributions from the community to make webnn-samples even better! If you have an idea for a new sample, an improvement to an existing one, or any other enhancement, please feel free to submit a pull request.

## Resources
### WebNN Resources
To learn more about Web Neural Network API (WebNN) and its capabilities, check out the following resources:
* [Web Neural Network API Specification](https://webmachinelearning.github.io/webnn/)
* [WebNN Polyfill](https://github.com/webmachinelearning/webnn-polyfill)
* [WebNN Community Group](https://webmachinelearning.github.io/)

### WebNN API Samples
* [WebNN code editor](https://webmachinelearning.github.io/webnn-samples/code/)
* [NNotepad](https://webmachinelearning.github.io/webnn-samples/nnotepad/)
* [Handwritten digits classification](https://webmachinelearning.github.io/webnn-samples/lenet/)
* Noise suppression:
  * [NSNet2](https://webmachinelearning.github.io/webnn-samples/nsnet2/)
  * [RNNoise](https://webmachinelearning.github.io/webnn-samples/rnnoise/)
* [Fast style transfer](https://webmachinelearning.github.io/webnn-samples/style_transfer/)
* [Semantic segmentation](https://webmachinelearning.github.io/webnn-samples/semantic_segmentation/)
* [Facial Landmark Detection](https://webmachinelearning.github.io/webnn-samples/facial_landmark_detection/)
* [Image classification](https://webmachinelearning.github.io/webnn-samples/image_classification/)
* [Object detection](https://webmachinelearning.github.io/webnn-samples/object_detection/)

## Acknowledgements
We thank the entire WebNN community for their valuable contributions and feedback. Your support and enthusiasm have been instrumental in making WebNN a robust and accessible tool for machine learning in the web ecosystem.

We also want to thank the developers and researchers behind the underlying technologies that power WebNN, including the Web Neural Network API and related frameworks. Their efforts have paved the way for seamless machine-learning experiences in web browsers.

We appreciate your interest in WebNN Samples! We hope you find these examples inspiring and educational. Happy coding with Web Neural Networks!

