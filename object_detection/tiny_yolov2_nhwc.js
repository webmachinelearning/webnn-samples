'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// Tiny Yolo V2 model with 'nhwc' layout, trained on the Pascal VOC dataset.
export class TinyYoloV2Nhwc {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = 'https://webmachinelearning.github.io/test-data/' +
        'models/tiny_yolov2_nhwc/weights/';
    this.inputOptions = {
      inputLayout: 'nhwc',
      labelUrl: './labels/pascal_classes.txt',
      margin: [1, 1, 1, 1],
      anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
      inputDimensions: [1, 416, 416, 3],
      norm: true,
    };
  }

  async buildConv_(input, name) {
    const prefix = this.weightsUrl_ + 'conv2d_' + name;
    const weightsName = prefix + '_kernel.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
      autoPad: 'same-upper',
    };
    return this.builder_.add(
        this.builder_.conv2d(input, weights, options),
        this.builder_.reshape(bias, [1, 1, 1, -1]));
  }

  async buildLeakyRelu_(input, name, maxPool2d = true, stride = 2) {
    const conv = await this.buildConv_(input, name);
    const alpha = this.builder_.constant({type: 'float32', dimensions: [1]},
        new Float32Array([0.10000000149011612]));
    const mul = this.builder_.mul(conv, alpha);
    const maximum = this.builder_.max(conv, mul);
    if (maxPool2d) {
      return this.builder_.maxPool2d(maximum, {
        windowDimensions: [2, 2],
        strides: [stride, stride],
        autoPad: 'same-upper',
        layout: 'nhwc',
      });
    }
    return maximum;
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const leakyRelu1 = await this.buildLeakyRelu_(input, '1');
    const leakyRelu2 = await this.buildLeakyRelu_(leakyRelu1, '2');
    const leakyRelu3 = await this.buildLeakyRelu_(leakyRelu2, '3');
    const leakyRelu4 = await this.buildLeakyRelu_(leakyRelu3, '4');
    const leakyRelu5 = await this.buildLeakyRelu_(leakyRelu4, '5');
    const leakyRelu6 = await this.buildLeakyRelu_(leakyRelu5, '6', true, 1);
    const leakyRelu7 = await this.buildLeakyRelu_(leakyRelu6, '7', false);
    const leakyRelu8 = await this.buildLeakyRelu_(leakyRelu7, '8', false);
    return await this.buildConv_(leakyRelu8, '9');
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
