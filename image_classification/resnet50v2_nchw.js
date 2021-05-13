'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// ResNet50 V2 model with 'nchw' input layout
export class ResNet50V2Nchw {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = 'https://webmachinelearning.github.io/test-data/' +
        'models/resnet50v2_nchw/weights/';
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
    };
  }

  async buildConv_(input, name, stageName, options = undefined) {
    let prefix = '';
    if (stageName !== '') {
      prefix = this.weightsUrl_ + 'resnetv24_stage' + stageName + '_conv' +
          name;
    } else {
      prefix = this.weightsUrl_ + 'resnetv24_conv' + name;
    }
    const weightName = prefix + '_weight.npy';
    const weight = await buildConstantByNpy(this.builder_, weightName);
    return this.builder_.conv2d(input, weight, options);
  }

  async buildBatchNorm_(input, name, stageName, relu = true) {
    let prefix = '';
    if (stageName !== '') {
      prefix = this.weightsUrl_ + 'resnetv24_stage' + stageName +
          '_batchnorm' + name;
    } else {
      prefix = this.weightsUrl_ + 'resnetv24_batchnorm' + name;
    }
    const scaleName = prefix + '_gamma.npy';
    const biasName = prefix + '_beta.npy';
    const meanName = prefix + '_running_mean.npy';
    const varName = prefix + '_running_var.npy';
    const scale = await buildConstantByNpy(this.builder_, scaleName);
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const mean = await buildConstantByNpy(this.builder_, meanName);
    const variance = await buildConstantByNpy(this.builder_, varName);
    const batchNorm = this.builder_.batchNormalization(
        input, mean, variance, {scale: scale, bias: bias});
    if (relu) {
      return this.builder_.relu(batchNorm);
    }
    return batchNorm;
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl_ + 'resnetv24_dense' + name;
    const weightName = prefix + '_weight.npy';
    const weight = await buildConstantByNpy(this.builder_, weightName);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const options = {c: bias, bTranspose: true};
    return this.builder_.gemm(input, weight, options);
  }

  async buildLinearBottleneck1_(input, stageName, stride = 1) {
    const strides = [stride, stride];

    const batchNorm0 = await this.buildBatchNorm_(input, '0', stageName);
    const conv0 = await this.buildConv_(batchNorm0, '0', stageName);
    const batchNorm1 = await this.buildBatchNorm_(conv0, '1', stageName);
    const conv3x3 = await this.buildConv_(
        batchNorm1, '1', stageName, {padding: [1, 1, 1, 1], strides});
    const batchNorm2 = await this.buildBatchNorm_(conv3x3, '2', stageName);
    const conv1 = await this.buildConv_(batchNorm2, '2', stageName);
    const conv2 = await this.buildConv_(batchNorm0, '3', stageName, {strides});

    return this.builder_.add(conv1, conv2);
  }

  async buildLinearBottleneck2_(input, stageName, nameIndex) {
    const batchNorm0 = await this.buildBatchNorm_(input, nameIndex, stageName);
    const conv0 = await this.buildConv_(batchNorm0, nameIndex + 1, stageName);
    const batchNorm1 = await this.buildBatchNorm_(
        conv0, nameIndex + 1, stageName);
    const conv3x3 = await this.buildConv_(
        batchNorm1, nameIndex + 2, stageName, {padding: [1, 1, 1, 1]});
    const batchNorm2 = await this.buildBatchNorm_(
        conv3x3, nameIndex + 2, stageName);
    const conv1 = await this.buildConv_(batchNorm2, nameIndex + 3, stageName);

    return this.builder_.add(input, conv1);
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    const data = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const batchNorm0 = await this.buildBatchNorm_(data, '0', '', false);
    const conv0 = await this.buildConv_(
        batchNorm0, '0', '', {pading: [3, 3, 3, 3], strides: [2, 2]});
    const batchNorm1 = await this.buildBatchNorm_(conv0, '1', '');
    const pool0 = await this.builder_.maxPool2d(batchNorm1,
        {windowDimensions: [3, 3], padding: [1, 1, 1, 1], strides: [2, 2]});

    // Stage 1
    const bottleneck0 = await this.buildLinearBottleneck1_(pool0, '1');
    const bottleneck1 = await this.buildLinearBottleneck2_(bottleneck0, '1', 3);
    const bottleneck2 = await this.buildLinearBottleneck2_(bottleneck1, '1', 6);

    // Stage 2
    const bottleneck3 = await this.buildLinearBottleneck1_(bottleneck2, '2', 2);
    const bottleneck4 = await this.buildLinearBottleneck2_(bottleneck3, '2', 3);
    const bottleneck5 = await this.buildLinearBottleneck2_(bottleneck4, '2', 6);
    const bottleneck6 = await this.buildLinearBottleneck2_(bottleneck5, '2', 9);

    // Stage 3
    const bottleneck7 = await this.buildLinearBottleneck1_(bottleneck6, '3', 2);
    const bottleneck8 = await this.buildLinearBottleneck2_(bottleneck7, '3', 3);
    const bottleneck9 = await this.buildLinearBottleneck2_(
        bottleneck8, '3', 6);
    const bottleneck10 = await this.buildLinearBottleneck2_(
        bottleneck9, '3', 9);
    const bottleneck11 = await this.buildLinearBottleneck2_(
        bottleneck10, '3', 12);
    const bottleneck12 = await this.buildLinearBottleneck2_(
        bottleneck11, '3', 15);

    // Stage 4
    const bottleneck13 = await this.buildLinearBottleneck1_(
        bottleneck12, '4', 2);
    const bottleneck14 = await this.buildLinearBottleneck2_(
        bottleneck13, '4', 3);
    const bottleneck15 = await this.buildLinearBottleneck2_(
        bottleneck14, '4', 6);

    const batchNorm2 = await this.buildBatchNorm_(bottleneck15, '2', '');
    const pool1 = await this.builder_.averagePool2d(batchNorm2);
    const reshape = this.builder_.reshape(pool1, [1, -1]);
    const gemm = await this.buildGemm_(reshape, '0');
    return this.builder_.softmax(gemm);
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if ('dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  async compute(inputBuffer) {
    const inputs = {input: {data: inputBuffer}};
    const outputs = await this.graph_.compute(inputs);
    return outputs;
  }
}
