'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// SimpleCNN model with 'nhwc' layout.
export class FaceLandmarkNhwc {
  constructor() {
    this.model_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/face_landmark_nhwc/weights';
    this.inputOptions = {
      inputLayout: 'nhwc',
      inputDimensions: [1, 128, 128, 3],
    };
  }

  async buildConv_(input, indice) {
    const prefix = `${this.weightsUrl_}/conv2d`;
    let weightSuffix = '_kernel.npy';
    let biasSuffix = `_Conv2D_bias.npy`;

    if (indice > 0) {
      weightSuffix = `_${indice}${weightSuffix}`;
      biasSuffix = `_${indice + 1}${biasSuffix}`;
    }

    const weightsName = prefix + weightSuffix;
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + biasSuffix;
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
      bias: bias,
      activation: this.builder_.relu(),
    };
    return this.builder_.conv2d(input, weights, options);
  }

  async buildFullyConnected_(input, namePrefix, relu = false, reshapeSize) {
    const weights = await buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/${namePrefix}_kernel_transpose.npy`);
    const bias = await buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/${namePrefix}_MatMul_bias.npy`);
    const options = {
      aTranspose: false,
      bTranspose: true,
      c: bias,
    };
    let fc;
    if (reshapeSize !== undefined) {
      fc = this.builder_.gemm(this.builder_.reshape(
          input, [-1, reshapeSize]), weights, options);
    } else {
      fc = this.builder_.gemm(input, weights, options);
    }
    if (relu) {
      fc = this.builder_.relu(fc);
    }

    return fc;
  }

  async load(contextOptions) {
    const context = navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(context);
    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});

    const poolOptions =
        {windowDimensions: [2, 2], strides: [2, 2], layout: 'nhwc'};

    const conv0 = await this.buildConv_(input, 0);
    const pool0 = await this.builder_.maxPool2d(conv0, poolOptions);

    const conv1 = await this.buildConv_(pool0, 1);
    const conv2 = await this.buildConv_(conv1, 2);
    const pool1 = await this.builder_.maxPool2d(conv2, poolOptions);

    const conv3 = await this.buildConv_(pool1, 3);
    const conv4 = await this.buildConv_(conv3, 4);
    const pool2 = await this.builder_.maxPool2d(conv4, poolOptions);

    const conv5 = await this.buildConv_(pool2, 5);
    const conv6 = await this.buildConv_(conv5, 6);
    const pool3 = await this.builder_.maxPool2d(
        conv6, {windowDimensions: [2, 2], layout: 'nhwc'});

    const conv7 = await this.buildConv_(pool3, 7);
    const fc0 = await this.buildFullyConnected_(
        conv7, 'dense', true, 6400);
    const fc1 = await this.buildFullyConnected_(fc0, 'logits');

    return fc1;
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  compute(inputBuffer, outputs) {
    const inputs = {'input': inputBuffer};
    this.graph_.compute(inputs, outputs);
  }
}
