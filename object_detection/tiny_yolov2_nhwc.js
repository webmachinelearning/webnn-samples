'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

// Tiny Yolo V2 model with 'nhwc' layout, trained on the Pascal VOC dataset.
export class TinyYoloV2Nhwc {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/tiny_yolov2_nhwc/weights/';
    this.inputOptions = {
      inputLayout: 'nhwc',
      labelUrl: './labels/pascal_classes.txt',
      margin: [1, 1, 1, 1],
      anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
      inputShape: [1, 416, 416, 3],
      norm: true,
    };
    this.outputShape_ = [1, 13, 13, 125];
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
    };
    options.bias = bias;
    options.padding = computePadding2DForAutoPad(
        /* nhwc */[input.shape[1], input.shape[2]],
        /* ohwi */[weights.shape[1], weights.shape[2]],
        options.strides, options.dilations, 'same-upper');
    let conv = this.builder_.conv2d(input, weights, options);
    if (leakyRelu) {
      // Fused leakyRelu is not supported by XNNPACK.
      conv = this.builder_.leakyRelu(conv, {alpha: 0.10000000149011612});
    }
    return conv;
  }

  buildMaxPool2d_(input, options) {
    options.padding = computePadding2DForAutoPad(
        /* nhwc */[input.shape[1], input.shape[2]],
        options.windowDimensions,
        options.strides, options.dilations, 'same-upper');
    return this.builder_.maxPool2d(input, options);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const inputDesc = {
      dataType: 'float32',
      dimensions: this.inputOptions.inputShape,
      shape: this.inputOptions.inputShape,
    };
    const input = this.builder_.input('input', inputDesc);
    inputDesc.usage = MLTensorUsage.WRITE;
    inputDesc.writable = true;
    this.inputTensor_ = await this.context_.createTensor(inputDesc);
    this.outputTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: this.outputShape_,
      shape: this.outputShape_,
      usage: MLTensorUsage.READ,
      readable: true,
    });

    const poolOptions = {
      windowDimensions: [2, 2],
      strides: [2, 2],
      layout: 'nhwc',
    };
    const conv1 = await this.buildConv_(input, '1');
    const pool1 = this.buildMaxPool2d_(conv1, poolOptions);
    const conv2 = await this.buildConv_(pool1, '2');
    const pool2 = this.buildMaxPool2d_(conv2, poolOptions);
    const conv3 = await this.buildConv_(pool2, '3');
    const pool3 = this.buildMaxPool2d_(conv3, poolOptions);
    const conv4 = await this.buildConv_(pool3, '4');
    const pool4 = this.buildMaxPool2d_(conv4, poolOptions);
    const conv5 = await this.buildConv_(pool4, '5');
    const pool5 = this.buildMaxPool2d_(conv5, poolOptions);
    const conv6 = await this.buildConv_(pool5, '6');
    const pool6 = this.buildMaxPool2d_(conv6,
        {windowDimensions: [2, 2], layout: 'nhwc'});
    const conv7 = await this.buildConv_(pool6, '7');
    const conv8 = await this.buildConv_(conv7, '8');
    return await this.buildConv_(conv8, '9', false);
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  async compute(inputBuffer) {
    this.context_.writeTensor(this.inputTensor_, inputBuffer);
    const inputs = {'input': this.inputTensor_};
    const outputs = {'output': this.outputTensor_};
    this.context_.dispatch(this.graph_, inputs, outputs);
    const results = await this.context_.readTensor(this.outputTensor_);
    return {'output': new Float32Array(results)};
  }
}
