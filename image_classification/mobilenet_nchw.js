'use strict';

import {buildConstantByNpy, weightsOrigin} from '../common/utils.js';

// MobileNet V2 model with 'nchw' input layout
export class MobileNetV2Nchw {
  constructor() {
    this.context_ = null;
    this.deviceType_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/mobilenetv2_nchw/weights/';
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

  async buildConv_(input, name, relu6 = true, options = {}) {
    const prefix = this.weightsUrl_ + 'conv_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias = buildConstantByNpy(this.builder_, biasName);
    options.bias = await bias;
    if (relu6) {
      // TODO: Set clamp activation to options once it's supported in
      // WebNN DML backend.
      // Implement `clip` by `clamp` of  WebNN API
      if (this.deviceType_ == 'gpu') {
        return this.builder_.clamp(
            this.builder_.conv2d(await input, await weights, options),
            {minValue: 0, maxValue: 6});
      } else {
        options.activation = this.builder_.clamp({minValue: 0, maxValue: 6});
      }
    }
    return this.builder_.conv2d(await input, await weights, options);
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl_ + 'gemm_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias = buildConstantByNpy(this.builder_, biasName);
    const options = {c: await bias, bTranspose: true};
    return this.builder_.gemm(await input, await weights, options);
  }

  async buildLinearBottleneck_(
      input, convNameArray, group, stride, shortcut = true) {
    const conv1x1Relu6 = this.buildConv_(await input, convNameArray[0]);
    const options = {
      padding: [1, 1, 1, 1],
      groups: group,
      strides: [stride, stride],
    };
    const dwise3x3Relu6 = this.buildConv_(
        await conv1x1Relu6, convNameArray[1], true, options);
    const conv1x1Linear = this.buildConv_(
        await dwise3x3Relu6, convNameArray[2], false);

    if (shortcut) {
      return this.builder_.add(await input, await conv1x1Linear);
    }
    return await conv1x1Linear;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.deviceType_ = contextOptions.deviceType;
    this.builder_ = new MLGraphBuilder(this.context_);
    const data = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });
    const conv0 = this.buildConv_(
        data, '0', true, {padding: [1, 1, 1, 1], strides: [2, 2]});
    const conv1 = this.buildConv_(
        conv0, '2', true, {padding: [1, 1, 1, 1], groups: 32});
    const conv2 = this.buildConv_(conv1, '4', false);
    const bottleneck0 = this.buildLinearBottleneck_(
        conv2, ['5', '7', '9'], 96, 2, false);
    const bottleneck1 = this.buildLinearBottleneck_(
        bottleneck0, ['10', '12', '14'], 144, 1);
    const bottleneck2 = this.buildLinearBottleneck_(
        bottleneck1, ['16', '18', '20'], 144, 2, false);
    const bottleneck3 = this.buildLinearBottleneck_(
        bottleneck2, ['21', '23', '25'], 192, 1);
    const bottleneck4 = this.buildLinearBottleneck_(
        bottleneck3, ['27', '29', '31'], 192, 1);
    const bottleneck5 = this.buildLinearBottleneck_(
        bottleneck4, ['33', '35', '37'], 192, 2, false);
    const bottleneck6 = this.buildLinearBottleneck_(
        bottleneck5, ['38', '40', '42'], 384, 1);
    const bottleneck7 = this.buildLinearBottleneck_(
        bottleneck6, ['44', '46', '48'], 384, 1);
    const bottleneck8 = this.buildLinearBottleneck_(
        bottleneck7, ['50', '52', '54'], 384, 1);
    const bottleneck9 = this.buildLinearBottleneck_(
        bottleneck8, ['56', '58', '60'], 384, 1, false);
    const bottleneck10 = this.buildLinearBottleneck_(
        bottleneck9, ['61', '63', '65'], 576, 1);
    const bottleneck11 = this.buildLinearBottleneck_(
        bottleneck10, ['67', '69', '71'], 576, 1);
    const bottleneck12 = this.buildLinearBottleneck_(
        bottleneck11, ['73', '75', '77'], 576, 2, false);
    const bottleneck13 = this.buildLinearBottleneck_(
        bottleneck12, ['78', '80', '82'], 960, 1);
    const bottleneck14 = this.buildLinearBottleneck_(
        bottleneck13, ['84', '86', '88'], 960, 1);
    const bottleneck15 = this.buildLinearBottleneck_(
        bottleneck14, ['90', '92', '94'], 960, 1, false);

    const conv3 = this.buildConv_(bottleneck15, '95', true);
    const pool = this.builder_.averagePool2d(await conv3);
    const reshape = this.builder_.reshape(pool, [1, 1280]);
    const gemm = this.buildGemm_(reshape, '104');
    return await this.builder_.softmax(await gemm);
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

  async compute(inputBuffer, outputBuffer) {
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results;
  }
}
