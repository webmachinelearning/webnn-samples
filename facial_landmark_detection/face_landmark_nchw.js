'use strict';

import {buildConstantByNpy, weightsOrigin} from '../common/utils.js';

// SimpleCNN model with 'nchw' layout.
export class FaceLandmarkNchw {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/face_landmark_nchw/weights';
    this.inputOptions = {
      inputLayout: 'nchw',
      inputDimensions: [1, 3, 128, 128],
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
      bias: await bias,
      activation: this.builder_.relu(),
    };
    return this.builder_.conv2d(await input, await weights, options);
  }

  async buildGemm_(input, namePrefix, relu = false, reshapeSize) {
    const weights = buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/${namePrefix}_kernel_transpose.npy`);
    const bias = buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/${namePrefix}_MatMul_bias.npy`);
    const options = {
      aTranspose: false,
      bTranspose: true,
      c: await bias,
    };
    let gemm;
    if (reshapeSize !== undefined) {
      gemm = this.builder_.gemm(this.builder_.reshape(
          this.builder_.transpose(await input, {permutation: [0, 2, 3, 1]}),
          [1, reshapeSize]), await weights, options);
    } else {
      gemm = this.builder_.gemm(await input, await weights, options);
    }
    if (relu) {
      gemm = this.builder_.relu(gemm);
    }

    return gemm;
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
        {windowDimensions: [2, 2], strides: [2, 2]};

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
    const pool3 = this.buildMaxPool2d(conv6, {windowDimensions: [2, 2]});

    const conv7 = this.buildConv_(pool3, 7);
    const gemm0 = this.buildGemm_(conv7, 'dense', true, 6400);
    const gemm1 = this.buildGemm_(gemm0, 'logits');

    return await gemm1;
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
