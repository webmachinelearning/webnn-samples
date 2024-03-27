'use strict';

import {buildConstantByNpy, weightsOrigin} from '../common/utils.js';

// ResNet50 V2 model with 'nchw' input layout
export class ResNet50V2Nchw {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/resnet50v2_nchw/weights/';
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
    };
    this.outputDimensions = [1, 1000];
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
    const weight = buildConstantByNpy(this.builder_, weightName);
    return this.builder_.conv2d(await input, await weight, options);
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
    const scale = buildConstantByNpy(this.builder_, scaleName);
    const bias = buildConstantByNpy(this.builder_, biasName);
    const mean = buildConstantByNpy(this.builder_, meanName);
    const variance = buildConstantByNpy(this.builder_, varName);
    const options = {scale: await scale, bias: await bias};
    if (relu) {
      options.activation = this.builder_.relu();
    }
    return this.builder_.batchNormalization(
        await input,
        await mean,
        await variance,
        options,
    );
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl_ + 'resnetv24_dense' + name;
    const weightName = prefix + '_weight.npy';
    const weight = buildConstantByNpy(this.builder_, weightName);
    const biasName = prefix + '_bias.npy';
    const bias = buildConstantByNpy(this.builder_, biasName);
    const options =
        {c: this.builder_.reshape(await bias, [1, 1000]), bTranspose: true};
    return this.builder_.gemm(await input, await weight, options);
  }

  async buildBottlenectV2_(
      input, stageName, nameIndices, downsample = false, stride = 1) {
    let residual = input;
    let strides = [1, 1];

    if (downsample) {
      strides = [stride, stride];
    }
    const bn1 = this.buildBatchNorm_(input, nameIndices[0], stageName);
    const conv1 = this.buildConv_(bn1, nameIndices[1], stageName);
    const bn2 = this.buildBatchNorm_(
        conv1, parseInt(nameIndices[0]) + 1, stageName);
    const conv2 = this.buildConv_(
        bn2, nameIndices[2], stageName, {padding: [1, 1, 1, 1], strides});
    const bn3 = this.buildBatchNorm_(
        conv2, parseInt(nameIndices[0]) + 2, stageName);
    const conv3 = this.buildConv_(bn3, nameIndices[3], stageName);
    if (downsample) {
      residual = this.buildConv_(
          bn1, parseInt(nameIndices[0]) + 3, stageName, {strides});
    }
    return this.builder_.add(await conv3, await residual);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const data = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });
    const bn1 = this.buildBatchNorm_(data, '0', '', false);
    const conv0 = this.buildConv_(
        bn1, '0', '', {padding: [3, 3, 3, 3], strides: [2, 2]});
    const bn2 = this.buildBatchNorm_(conv0, '1', '');
    const pool1 = this.builder_.maxPool2d(await bn2,
        {windowDimensions: [3, 3], padding: [1, 1, 1, 1], strides: [2, 2]});

    // Stage 1
    const bottleneck1 = this.buildBottlenectV2_(
        pool1, '1', ['0', '0', '1', '2'], true);
    const bottleneck2 = this.buildBottlenectV2_(
        bottleneck1, '1', ['3', '4', '5', '6']);
    const bottleneck3 = this.buildBottlenectV2_(
        bottleneck2, '1', ['6', '7', '8', '9']);

    // Stage 2
    const bottleneck4 = this.buildBottlenectV2_(
        bottleneck3, '2', ['0', '0', '1', '2'], true, 2);
    const bottleneck5 = this.buildBottlenectV2_(
        bottleneck4, '2', ['3', '4', '5', '6']);
    const bottleneck6 = this.buildBottlenectV2_(
        bottleneck5, '2', ['6', '7', '8', '9']);
    const bottleneck7 = this.buildBottlenectV2_(
        bottleneck6, '2', ['9', '10', '11', '12']);

    // Stage 3
    const bottleneck8 = this.buildBottlenectV2_(
        bottleneck7, '3', ['0', '0', '1', '2'], true, 2);
    const bottleneck9 = this.buildBottlenectV2_(
        bottleneck8, '3', ['3', '4', '5', '6']);
    const bottleneck10 = this.buildBottlenectV2_(
        bottleneck9, '3', ['6', '7', '8', '9']);
    const bottleneck11 = this.buildBottlenectV2_(
        bottleneck10, '3', ['9', '10', '11', '12']);
    const bottleneck12 = this.buildBottlenectV2_(
        bottleneck11, '3', ['12', '13', '14', '15']);
    const bottleneck13 = this.buildBottlenectV2_(
        bottleneck12, '3', ['15', '16', '17', '18']);

    // Stage 4
    const bottleneck14 = this.buildBottlenectV2_(
        bottleneck13, '4', ['0', '0', '1', '2'], true, 2);
    const bottleneck15 = this.buildBottlenectV2_(
        bottleneck14, '4', ['3', '4', '5', '6']);
    const bottleneck16 = this.buildBottlenectV2_(
        bottleneck15, '4', ['6', '7', '8', '9']);

    const bn3 = this.buildBatchNorm_(bottleneck16, '2', '');
    const pool2 = this.builder_.averagePool2d(await bn3);
    const reshape = this.builder_.reshape(await pool2, [1, 2048]);
    const gemm = this.buildGemm_(await reshape, '0');
    return this.builder_.softmax(await gemm);
  }

  async build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  // Release the constant tensors of a model
  async compute(inputBuffer, outputBuffer) {
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    const results = await this.context_.compute(
        await this.graph_,
        inputs,
        outputs,
    );
    return results;
  }
}
