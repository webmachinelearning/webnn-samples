'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// Tiny Yolo V2 model with 'nchw' layout, trained on the Pascal VOC dataset.
export class TinyYoloV2Nchw {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = 'https://webmachinelearning.github.io/test-data/' +
        'models/tiny_yolov2_nchw/weights/';
    this.inputOptions = {
      inputLayout: 'nchw',
      labelUrl: './labels/pascal_classes.txt',
      margin: [1, 1, 1, 1],
      anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
      inputDimensions: [1, 3, 416, 416],
    };
  }

  async buildConv_(input, name, useBias = false) {
    const prefix = this.weightsUrl_ + 'convolution' + name;
    const weightName = prefix + '_W.npy';
    const weight = await buildConstantByNpy(this.builder_, weightName);
    const conv = this.builder_.conv2d(input, weight, {autoPad: 'same-upper'});

    if (useBias) {
      const biasName = prefix + '_B.npy';
      const bias = await buildConstantByNpy(this.builder_, biasName);
      return this.builder_.add(
          conv, this.builder_.reshape(bias, [1, -1, 1, 1]));
    }
    return conv;
  }

  async buildBatchNorm_(input, name) {
    const prefix = this.weightsUrl_ + 'BatchNormalization';
    const scaleName = `${prefix}_scale${name}.npy`;
    const biasName = `${prefix}_B${name}.npy`;
    const meanName = `${prefix}_mean${name}.npy`;
    const varName = `${prefix}_variance${name}.npy`;
    const scale = await buildConstantByNpy(this.builder_, scaleName);
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const mean = await buildConstantByNpy(this.builder_, meanName);
    const variance = await buildConstantByNpy(this.builder_, varName);

    const batchNorm = this.builder_.batchNormalization(
        input, mean, variance, {scale: scale, bias: bias});
    return batchNorm;
  }

  async buildLeakyRelu_(input, name, maxPool2d = true, stride = 2) {
    const conv = await this.buildConv_(input, name);
    const batchNorm = await this.buildBatchNorm_(conv, name);
    const leakyRelu =
        this.builder_.leakyRelu(batchNorm, {alpha: 0.10000000149011612});

    if (maxPool2d) {
      return this.builder_.maxPool2d(leakyRelu, {
        windowDimensions: [2, 2],
        strides: [stride, stride],
        autoPad: 'same-upper',
      });
    }
    return leakyRelu;
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    const image = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});

    const mulScale = this.builder_.constant({type: 'float32',
      dimensions: [1]}, new Float32Array([0.003921568859368563]));
    const addBias = this.builder_.constant({type: 'float32',
      dimensions: [3, 1, 1]}, new Float32Array([0, 0, 0]));

    const mul = this.builder_.mul(image, mulScale);
    const add = this.builder_.add(mul, addBias);
    const loop0 = await this.buildLeakyRelu_(add, '');
    const loop1 = await this.buildLeakyRelu_(loop0, '1');
    const loop2 = await this.buildLeakyRelu_(loop1, '2');
    const loop3 = await this.buildLeakyRelu_(loop2, '3');
    const loop4 = await this.buildLeakyRelu_(loop3, '4');
    const loop5 = await this.buildLeakyRelu_(loop4, '5', true, 1);
    const loop6 = await this.buildLeakyRelu_(loop5, '6', false);
    const loop7 = await this.buildLeakyRelu_(loop6, '7', false);
    const conv = await this.buildConv_(loop7, '8', true);
    return conv;
    // return this.builder_.transpose(conv, {permutation: [0, 2, 3, 1]});
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  async compute(inputBuffer) {
    const inputs = {input: {data: inputBuffer}};
    const outputs = await this.graph_.compute(inputs);
    return outputs;
  }
}
