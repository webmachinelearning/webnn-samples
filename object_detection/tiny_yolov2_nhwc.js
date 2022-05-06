'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// Tiny Yolo V2 model with 'nhwc' layout, trained on the Pascal VOC dataset.
export class TinyYoloV2Nhwc {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/tiny_yolov2_nhwc/weights/';
    this.inputOptions = {
      inputLayout: 'nhwc',
      labelUrl: './labels/pascal_classes.txt',
      margin: [1, 1, 1, 1],
      anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
      inputDimensions: [1, 416, 416, 3],
      norm: true,
    };
    this.outputDimensions = [1, 13, 13, 125];
  }

  async buildConv_(input, name, leakyRelu = true) {
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
    options.bias = bias;
    if (leakyRelu) {
      options.activation =
          this.builder_.leakyRelu({alpha: 0.10000000149011612});
    }
    return this.builder_.conv2d(input, weights, options);
  }

  async load(contextOptions) {
    const context = navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(context);
    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});

    const poolOptions = {
      windowDimensions: [2, 2],
      strides: [2, 2],
      autoPad: 'same-upper',
      layout: 'nhwc',
    };
    const conv1 = await this.buildConv_(input, '1');
    const pool1 = this.builder_.maxPool2d(conv1, poolOptions);
    const conv2 = await this.buildConv_(pool1, '2');
    const pool2 = this.builder_.maxPool2d(conv2, poolOptions);
    const conv3 = await this.buildConv_(pool2, '3');
    const pool3 = this.builder_.maxPool2d(conv3, poolOptions);
    const conv4 = await this.buildConv_(pool3, '4');
    const pool4 = this.builder_.maxPool2d(conv4, poolOptions);
    const conv5 = await this.buildConv_(pool4, '5');
    const pool5 = this.builder_.maxPool2d(conv5, poolOptions);
    const conv6 = await this.buildConv_(pool5, '6');
    const pool6 = this.builder_.maxPool2d(conv6,
        {windowDimensions: [2, 2], autoPad: 'same-upper', layout: 'nhwc'});
    const conv7 = await this.buildConv_(pool6, '7');
    const conv8 = await this.buildConv_(conv7, '8');
    return await this.buildConv_(conv8, '9', false);
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

  async compute(inputBuffer, outputs) {
    const inputs = {'input': inputBuffer};
    await this.graph_.compute(inputs, outputs);
  }
}
