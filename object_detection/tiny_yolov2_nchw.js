'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

// Tiny Yolo V2 model with 'nchw' layout, trained on the Pascal VOC dataset.
export class TinyYoloV2Nchw {
  constructor(dataType = 'float32') {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.deviceType_ = null;
    this.targetDataType_ = dataType;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/tiny_yolov2_nchw/weights/';
    this.inputOptions = {
      inputLayout: 'nchw',
      labelUrl: './labels/pascal_classes.txt',
      margin: [1, 1, 1, 1],
      anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
      inputShape: [1, 3, 416, 416],
    };
    this.outputShape_ = [1, 13, 13, 125];
  }

  async buildConv_(input, name) {
    let biasName =
        `${this.weightsUrl_}ConvBnFusion_BN_B_BatchNormalization_B${name}.npy`;
    let weightName =
        `${this.weightsUrl_}ConvBnFusion_W_convolution${name}_W.npy`;
    if (name === '8') {
      biasName = `${this.weightsUrl_}convolution8_B.npy`;
      weightName = `${this.weightsUrl_}convolution8_W.npy`;
    }

    const weight = await buildConstantByNpy(
        this.builder_, weightName, this.targetDataType_);
    const options = {autoPad: 'same-upper'};
    options.padding = computePadding2DForAutoPad(
        /* nchw */[input.shape()[2], input.shape()[3]],
        /* oihw */[weight.shape()[2], weight.shape()[3]],
        options.strides, options.dilations, 'same-upper');
    options.bias = await buildConstantByNpy(
        this.builder_, biasName, this.targetDataType_);
    const conv = this.builder_.conv2d(input, weight, options);
    if (name === '8') {
      return conv;
    } else {
      return this.builder_.leakyRelu(conv, {alpha: 0.10000000149011612});
    }
  }

  buildMaxPool2d_(input, options) {
    options.padding = computePadding2DForAutoPad(
        /* nchw */[input.shape()[2], input.shape()[3]],
        options.windowDimensions,
        options.strides, options.dilations, 'same-upper');
    return this.builder_.maxPool2d(input, options);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.deviceType_ = contextOptions.deviceType;
    this.builder_ = new MLGraphBuilder(this.context_);
    const inputDesc = {
      dataType: 'float32',
      dimensions: this.inputOptions.inputShape,
      shape: this.inputOptions.inputShape,
    };
    let image = this.builder_.input('input', inputDesc);
    inputDesc.usage = MLTensorUsage.WRITE;
    this.inputTensor_ = await this.context_.createTensor(inputDesc);
    this.outputTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: this.outputShape_,
      shape: this.outputShape_,
      usage: MLTensorUsage.READ,
    });

    let mulScale = this.builder_.constant(
        {dataType: 'float32', dimensions: [1], shape: [1]},
        new Float32Array([0.003921568859368563]),
    );
    const poolOptions = {
      windowDimensions: [2, 2],
      strides: [2, 2],
    };
    if (this.targetDataType_ === 'float16') {
      image = this.builder_.cast(image, 'float16');
      mulScale = this.builder_.cast(mulScale, 'float16');
    }
    const mul = this.builder_.mul(image, mulScale);
    const conv0 = await this.buildConv_(mul, '');
    const pool0 = this.buildMaxPool2d_(conv0, poolOptions);
    const conv1 = await this.buildConv_(pool0, '1');
    const pool1 = this.buildMaxPool2d_(conv1, poolOptions);
    const conv2 = await this.buildConv_(pool1, '2');
    const pool2 = this.buildMaxPool2d_(conv2, poolOptions);
    const conv3 = await this.buildConv_(pool2, '3');
    const pool3 = this.buildMaxPool2d_(conv3, poolOptions);
    const conv4 = await this.buildConv_(pool3, '4');
    const pool4 = this.buildMaxPool2d_(conv4, poolOptions);
    const conv5 = await this.buildConv_(pool4, '5');
    const pool5 = this.buildMaxPool2d_(conv5, {windowDimensions: [2, 2]});
    const conv6 = await this.buildConv_(pool5, '6');
    const conv7 = await this.buildConv_(conv6, '7');
    const conv = await this.buildConv_(conv7, '8');
    const transpose = this.builder_.transpose(
        conv, {permutation: [0, 2, 3, 1]});
    if (this.targetDataType_ === 'float16') {
      return this.builder_.cast(transpose, 'float32');
    } else {
      return transpose;
    }
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
