'use strict';

import {buildConstantByNpy, weightsOrigin} from '../common/utils.js';

// SimpleCNN model with 'nhwc' layout.
export class FaceLandmarkNhwc {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/face_landmark_nhwc/weights';
    this.inputOptions = {
      inputLayout: 'nhwc',
      inputDimensions: [1, 128, 128, 3],
    };
  }

  async buildMaxPool2d(input, options) {
    return this.builder_.maxPool2d(await input, options);
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
    const weights = buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + biasSuffix;
    const bias = buildConstantByNpy(this.builder_, biasName);
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
      bias: await bias,
      activation: this.builder_.relu(),
    };
    return this.builder_.conv2d(await input, await weights, options);
  }

  async buildFullyConnected_(input, namePrefix, relu = false, reshapeSize) {
    const weights = buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/${namePrefix}_kernel_transpose.npy`);
    const bias = buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/${namePrefix}_MatMul_bias.npy`);
    const options = {
      aTranspose: false,
      bTranspose: true,
      c: await bias,
    };
    let fc;
    if (reshapeSize !== undefined) {
      fc = this.builder_.gemm(this.builder_.reshape(
          await input, [1, reshapeSize]), await weights, options);
    } else {
      fc = this.builder_.gemm(await input, await weights, options);
    }
    if (relu) {
      fc = this.builder_.relu(fc);
    }

    return fc;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const input = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });

    const poolOptions =
        {windowDimensions: [2, 2], strides: [2, 2], layout: 'nhwc'};

    const conv0 = this.buildConv_(input, 0);
    const pool0 = this.buildMaxPool2d(conv0, poolOptions);

    const conv1 = this.buildConv_(pool0, 1);
    const conv2 = this.buildConv_(conv1, 2);
    const pool1 = this.buildMaxPool2d(conv2, poolOptions);

    const conv3 = this.buildConv_(pool1, 3);
    const conv4 = this.buildConv_(conv3, 4);
    const pool2 = this.buildMaxPool2d(conv4, poolOptions);

    const conv5 = this.buildConv_(pool2, 5);
    const conv6 = this.buildConv_(conv5, 6);
    const pool3 = this.buildMaxPool2d(
        conv6, {windowDimensions: [2, 2], layout: 'nhwc'});

    const conv7 = this.buildConv_(pool3, 7);
    const fc0 = this.buildFullyConnected_(
        conv7, 'dense', true, 6400);
    const fc1 = this.buildFullyConnected_(fc0, 'logits');

    return await fc1;
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
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results;
  }
}
