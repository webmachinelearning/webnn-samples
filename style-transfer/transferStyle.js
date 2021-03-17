import { StyleTransfer } from './styleTransfer.js';

// Operate Style Transfer Model
export class TransferStyle {
  constructor(inputElement) {
    this.styleTransfer_ = new StyleTransfer();
    this.inputElement_ = inputElement;
    this.inputSize_ = [];
    this.inferenceTime_ = 0;
  }

  async load(modelId) {
    console.log(`- Model ID: ${modelId} -`);
    console.log('- Loading weights... ');
    let start = performance.now();
    await this.styleTransfer_.load(`./weights/${modelId}/`);
    const modelLoadTime = (performance.now() - start).toFixed(2);
    console.log(`  done in ${modelLoadTime} ms.`);
  }

  async compile() {
    console.log('- Compiling... ');
    let start = performance.now();
    await this.styleTransfer_.compile();
    const modelCompileTime = (performance.now() - start).toFixed(2);
    console.log(`  done in ${modelCompileTime} ms.`);
  }

  async compute() {
    const inputData = this.getInputTensor_();
    console.log('- Computing... ');
    let start = performance.now();
    const outputs = await this.styleTransfer_.compute(inputData);
    this.inferenceTime_ = (performance.now() - start).toFixed(2);
    console.log(`  done in ${this.inferenceTime_} ms.`);
    return outputs;
  }

  getInferenceTime() {
    return this.inferenceTime_;
  }

  // Covert input element to tensor data
  getInputTensor_() {
    this.inputSize_ = this.styleTransfer_.getDimensions();
    let tensor = new Float32Array(this.inputSize_.slice(1).reduce((a, b) => a * b));

    this.inputElement_.width = this.inputElement_.videoWidth || this.inputElement_.naturalWidth;
    this.inputElement_.height = this.inputElement_.videoHeight || this.inputElement_.naturalHeight;

    let [channels, height, width] = this.inputSize_.slice(1);
    const mean = [0, 0, 0, 0];
    const std = [1, 1, 1, 1];
    const imageChannels = 4; // RGBA

    let canvasElement = document.createElement('canvas');
    canvasElement.width = width;
    canvasElement.height = height;
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(this.inputElement_, 0, 0, width, height);

    let pixels = canvasContext.getImageData(0, 0, width, height).data;

    for (let c = 0; c < channels; ++c) {
      for (let h = 0; h < height; ++h) {
        for (let w = 0; w < width; ++w) {
          let value = pixels[h * width * imageChannels + w * imageChannels + c];
          tensor[c * width * height + h * width + w] = (value - mean[c]) / std[c];
        }
      }
    }
    return tensor;
  }
}
