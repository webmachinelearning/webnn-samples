'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';
const strides = [2, 2];
const autoPad = 'same-upper';

/* eslint-disable camelcase */

// FaceNet model with 'nhwc' layout.
export class FaceNetNhwc {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/facenet_nhwc/weights';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5, 127.5],
      channelScheme: 'BGR',
      inputLayout: 'nhwc',
      inputShape: [1, 160, 160, 3],
    };
    this.postOptions = {
      distanceMetric: 'euclidean',
      threshold: 1.26,
    };
    this.outputShape_ = [1, 512];
  }

  async buildConv_(input, namePrefix, options = undefined, relu = true) {
    const weightsName = `${this.weightsUrl_}/${namePrefix}_kernel.npy`;
    const biasName = `${this.weightsUrl_}/${namePrefix}_Conv2D_bias.npy`;
    const [weights, bias] = await Promise.all([
      weightsName, biasName,
    ].map((name) => buildConstantByNpy(this.builder_, name)));
    if (options !== undefined) {
      options.inputLayout = 'nhwc';
      options.filterLayout = 'ohwi';
      options.bias = bias;
    } else {
      options = {
        inputLayout: 'nhwc',
        filterLayout: 'ohwi',
        bias: bias,
      };
    }

    input = await input;

    // WebNN spec drops autoPad support, compute the explicit padding instead.
    if (options.autoPad == 'same-upper') {
      options.padding =
        computePadding2DForAutoPad(
            /* nwhc */[input.shape()[1], input.shape()[2]],
            /* ohwi */[weights.shape()[1], weights.shape()[2]],
            options.strides, options.dilations, options.autoPad);
    }
    const conv2d = this.builder_.conv2d(input, weights, options);
    return relu ? this.builder_.relu(conv2d) : conv2d;
  }

  async buildBlock35_(input, indice) {
    const branch0 = this.buildConv_(
        input, `Block35_${indice}_Branch_0_Conv2d_1x1`, {autoPad});
    const branch1_0 = this.buildConv_(
        input, `Block35_${indice}_Branch_1_Conv2d_0a_1x1`, {autoPad});
    const branch1_1 = this.buildConv_(
        branch1_0, `Block35_${indice}_Branch_1_Conv2d_0b_3x3`, {autoPad});
    const branch2_0 = this.buildConv_(
        input, `Block35_${indice}_Branch_2_Conv2d_0a_1x1`, {autoPad});
    const branch2_1 = this.buildConv_(
        branch2_0, `Block35_${indice}_Branch_2_Conv2d_0b_3x3`, {autoPad});
    const branch2_2 = this.buildConv_(
        branch2_1, `Block35_${indice}_Branch_2_Conv2d_0c_3x3`, {autoPad});

    const concat = Promise.all([branch0, branch1_1, branch2_2])
        .then((inputs) => this.builder_.concat(inputs, 3));
    const conv = this.buildConv_(
        concat, `Block35_${indice}_Conv2d_1x1`, {autoPad}, false);

    return this.builder_.relu(this.builder_.add(await input, await conv));
  }

  async buildBlock17_(input, indice) {
    const branch0 = this.buildConv_(
        input, `Block17_${indice}_Branch_0_Conv2d_1x1`, {autoPad});
    const branch1_0 = this.buildConv_(
        input, `Block17_${indice}_Branch_1_Conv2d_0a_1x1`, {autoPad});
    const branch1_1 = this.buildConv_(
        branch1_0, `Block17_${indice}_Branch_1_Conv2d_0b_1x7`, {autoPad});
    const branch1_2 = this.buildConv_(
        branch1_1, `Block17_${indice}_Branch_1_Conv2d_0c_7x1`, {autoPad});

    const concat = Promise.all([branch0, branch1_2])
        .then((inputs) => this.builder_.concat(inputs, 3));
    const conv = this.buildConv_(
        concat, `Block17_${indice}_Conv2d_1x1`, {autoPad}, false);

    return this.builder_.relu(this.builder_.add(await input, await conv));
  }

  async buildBlock8_(input, indice, relu = true) {
    const branch0 = this.buildConv_(
        input, `Block8_${indice}_Branch_0_Conv2d_1x1`, {autoPad});
    const branch1_0 = this.buildConv_(
        input, `Block8_${indice}_Branch_1_Conv2d_0a_1x1`, {autoPad});
    const branch1_1 = this.buildConv_(
        branch1_0, `Block8_${indice}_Branch_1_Conv2d_0b_1x3`, {autoPad});
    const branch1_2 = this.buildConv_(
        branch1_1, `Block8_${indice}_Branch_1_Conv2d_0c_3x1`, {autoPad});

    const concat = Promise.all([branch0, branch1_2])
        .then((inputs) => this.builder_.concat(inputs, 3));
    const conv = this.buildConv_(
        concat, `Block8_${indice}_Conv2d_1x1`, {autoPad}, false);

    let result = this.builder_.add(await input, await conv);

    if (relu) {
      result = this.builder_.relu(result);
    }
    return result;
  }

  async buildFullyConnected_(input) {
    const weights = buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/Bottleneck_kernel_transpose.npy`);
    const bias = buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/Bottleneck_MatMul_bias.npy`);
    input = this.builder_.reshape(await input, [1, 1792]);
    const options = {
      aTranspose: false,
      bTranspose: true,
      c: await bias,
    };
    return this.builder_.gemm(input, await weights, options);
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
    this.inputTensor_ = await this.context_.createTensor(inputDesc);
    this.outputTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: this.outputShape_,
      shape: this.outputShape_,
      usage: MLTensorUsage.READ,
    });

    const poolOptions = {windowDimensions: [3, 3], strides, layout: 'nhwc'};

    const conv0 = this.buildConv_(input, 'Conv2d_1a_3x3', {strides});
    const conv1 = this.buildConv_(conv0, 'Conv2d_2a_3x3');
    const conv2 = this.buildConv_(conv1, 'Conv2d_2b_3x3', {autoPad});

    const pool0 = conv2.then((conv2) =>
      this.builder_.maxPool2d(conv2, poolOptions));

    const conv3 = this.buildConv_(pool0, 'Conv2d_3b_1x1');
    const conv4 = this.buildConv_(conv3, 'Conv2d_4a_3x3');
    const conv5 = this.buildConv_(conv4, 'Conv2d_4b_3x3', {strides});

    // Block 35
    const block35_1 = this.buildBlock35_(conv5, 1);
    const block35_2 = this.buildBlock35_(block35_1, 2);
    const block35_3 = this.buildBlock35_(block35_2, 3);
    const block35_4 = this.buildBlock35_(block35_3, 4);
    const block35_5 = this.buildBlock35_(block35_4, 5);

    // Mixed 6a branches
    const mixed6a_branch0 = this.buildConv_(
        block35_5, 'Mixed_6a_Branch_0_Conv2d_1a_3x3', {strides});
    const mixed6a_pool = block35_5.then((block35_5) =>
      this.builder_.maxPool2d(block35_5, poolOptions));
    const mixed6a_branch1_0 = this.buildConv_(
        block35_5, 'Mixed_6a_Branch_1_Conv2d_0a_1x1', {autoPad});
    const mixed6a_branch1_1 = this.buildConv_(
        mixed6a_branch1_0, 'Mixed_6a_Branch_1_Conv2d_0b_3x3', {autoPad});
    const mixed6a_branch1_2 = this.buildConv_(
        mixed6a_branch1_1, 'Mixed_6a_Branch_1_Conv2d_1a_3x3', {strides});
    const mixed6a = Promise.all([
      mixed6a_branch0, mixed6a_branch1_2, mixed6a_pool,
    ]).then((inputs) => this.builder_.concat(inputs, 3));

    // Block 17
    const block17_1 = this.buildBlock17_(mixed6a, 1);
    const block17_2 = this.buildBlock17_(block17_1, 2);
    const block17_3 = this.buildBlock17_(block17_2, 3);
    const block17_4 = this.buildBlock17_(block17_3, 4);
    const block17_5 = this.buildBlock17_(block17_4, 5);
    const block17_6 = this.buildBlock17_(block17_5, 6);
    const block17_7 = this.buildBlock17_(block17_6, 7);
    const block17_8 = this.buildBlock17_(block17_7, 8);
    const block17_9 = this.buildBlock17_(block17_8, 9);
    const block17_10 = this.buildBlock17_(block17_9, 10);

    // Mixed 7a branches
    const mixed7a_pool = block17_10.then((block17_10) =>
      this.builder_.maxPool2d(block17_10, poolOptions));
    const mixed7a_branch0_0 = this.buildConv_(
        block17_10, 'Mixed_7a_Branch_0_Conv2d_0a_1x1', {autoPad});
    const mixed7a_branch0_1 = this.buildConv_(
        mixed7a_branch0_0, 'Mixed_7a_Branch_0_Conv2d_1a_3x3', {strides});
    const mixed7a_branch1_0 = this.buildConv_(
        block17_10, 'Mixed_7a_Branch_1_Conv2d_0a_1x1', {autoPad});
    const mixed7a_branch1_1 = this.buildConv_(
        mixed7a_branch1_0, 'Mixed_7a_Branch_1_Conv2d_1a_3x3', {strides});
    const mixed7a_branch2_0 = this.buildConv_(
        block17_10, 'Mixed_7a_Branch_2_Conv2d_0a_1x1', {autoPad});
    const mixed7a_branch2_1 = this.buildConv_(
        mixed7a_branch2_0, 'Mixed_7a_Branch_2_Conv2d_0b_3x3', {autoPad});
    const mixed7a_branch2_2 = this.buildConv_(
        mixed7a_branch2_1, 'Mixed_7a_Branch_2_Conv2d_1a_3x3', {strides});
    const mixed7a = Promise.all([
      mixed7a_branch0_1, mixed7a_branch1_1, mixed7a_branch2_2,
      mixed7a_pool,
    ]).then((inputs) => this.builder_.concat(inputs, 3));

    // Block 8
    const block8_1 = this.buildBlock8_(mixed7a, 1);
    const block8_2 = this.buildBlock8_(block8_1, 2);
    const block8_3 = this.buildBlock8_(block8_2, 3);
    const block8_4 = this.buildBlock8_(block8_3, 4);
    const block8_5 = this.buildBlock8_(block8_4, 5);
    const block8_6 = this.buildBlock8_(block8_5, 6, false);

    const mean = block8_6.then((block8_6) =>
      this.builder_.averagePool2d(block8_6, {layout: 'nhwc'}));
    const fc = await this.buildFullyConnected_(mean);
    // L2Normalization will be handled in post-processing
    return fc;
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
    return new Float32Array(results);
  }
}
