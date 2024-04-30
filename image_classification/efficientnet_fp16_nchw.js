'use strict';

import {buildConstantByNpy, weightsOrigin} from '../common/utils.js';

// EfficientNet fp16 model with 'nchw' input layout
export class EfficientNetFP16Nchw {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
    '/test-data/models/efficientnet_fp16_nchw_optimized/weights/';
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
      dataType: 'float16',
    };
    this.outputDimensions = [1, 1000];
  }

  async buildConv_(input, name, blockName, clip = false, options = {}) {
    let prefix = '';
    if (blockName !== '') {
      prefix = this.weightsUrl_ + 'block' + blockName + '_conv' +
          name;
    } else {
      prefix = this.weightsUrl_ + 'conv' + name;
    }
    const weight = buildConstantByNpy(this.builder_, prefix + '_w.npy');
    options.bias = await buildConstantByNpy(this.builder_, prefix + '_b.npy');
    if (clip) {
      return this.builder_.clamp(
          this.builder_.conv2d(await input, await weight, options),
          {minValue: 0, maxValue: 6});
    }
    return this.builder_.conv2d(await input, await weight, options);
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl_ + 'dense' + name;
    const weightName = prefix + '_w.npy';
    const weight = buildConstantByNpy(this.builder_, weightName);
    const biasName = prefix + '_b.npy';
    const bias = buildConstantByNpy(this.builder_, biasName);
    const options =
        {c: this.builder_.reshape(await bias, [1, 1000])};
    return this.builder_.gemm(await input, await weight, options);
  }

  async buildBottleneck_(input, blockName, group, pad = 1) {
    const conv1 = this.buildConv_(input, '0', blockName, true);
    const conv2 = this.buildConv_(conv1, '1', blockName, true,
        {groups: group, padding: [pad, pad, pad, pad]});
    const conv3 = this.buildConv_(conv2, '2', blockName);
    return this.builder_.add(await conv3, await input);
  }

  async buildBottlenecks_(input, blockNames, group, pad = 1) {
    let result = input;
    for (let i = 0; i < blockNames.length; i++) {
      const bottleneck = await this.buildBottleneck_(result, blockNames[i],
          group, pad);
      result = bottleneck;
    }
    return result;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const data = this.builder_.input('input', {
      dataType: this.inputOptions.dataType,
      dimensions: this.inputOptions.inputDimensions,
    });
    // Block 0
    const conv1 = this.buildConv_(
        data, '0', '0', true, {padding: [0, 1, 0, 1], strides: [2, 2]});
    const conv2 = this.buildConv_(conv1, '1', '0', true,
        {groups: 32, padding: [1, 1, 1, 1]});
    const conv3 = this.buildConv_(conv2, '2', '0');

    // Block 1
    const conv4 = this.buildConv_(conv3, '0', '1', true);
    const conv5 = this.buildConv_(conv4, '1', '1', true,
        {groups: 144, padding: [0, 1, 0, 1], strides: [2, 2]});
    const conv6 = this.buildConv_(conv5, '2', '1');

    // Block 2~4
    const bottleneck4 = this.buildBottlenecks_(conv6,
        ['2', '3', '4'], 192);

    // Block 5
    const conv7 = this.buildConv_(bottleneck4, '0', '5', true);
    const conv8 = this.buildConv_(conv7, '1', '5', true,
        {groups: 192, padding: [1, 2, 1, 2], strides: [2, 2]});
    const conv9 = this.buildConv_(conv8, '2', '5');

    // Block 6~8
    const bottleneck8 = this.buildBottlenecks_(conv9,
        ['6', '7', '8'], 336, 2);

    // Block 9
    const conv10 = this.buildConv_(bottleneck8, '0', '9', true);
    const conv11 = this.buildConv_(conv10, '1', '9', true,
        {groups: 336, padding: [0, 1, 0, 1], strides: [2, 2]});
    const conv12 = this.buildConv_(conv11, '2', '9');

    // Block 10~14
    const bottleneck14 = this.buildBottlenecks_(conv12,
        ['10', '11', '12', '13', '14'], 672);

    // Block 15
    const conv13 = this.buildConv_(bottleneck14, '0', '15', true);
    const conv14 = this.buildConv_(conv13, '1', '15', true,
        {groups: 672, padding: [2, 2, 2, 2]});
    const conv15 = this.buildConv_(conv14, '2', '15');

    // Block 16~20
    const bottleneck20 = await this.buildBottlenecks_(conv15,
        ['16', '17', '18', '19', '20'], 960, 2);

    // Block 21
    const conv16 = this.buildConv_(bottleneck20, '0', '21', true);
    const conv17 = this.buildConv_(conv16, '1', '21', true,
        {groups: 960, padding: [1, 2, 1, 2], strides: [2, 2]});
    const conv18 = this.buildConv_(conv17, '2', '21');

    // Block 22~28
    const bottleneck28 = this.buildBottlenecks_(conv18,
        ['22', '23', '24', '25', '26', '27', '28'], 1632, 2);

    // Block 29
    const conv19 = this.buildConv_(bottleneck28, '0', '29', true);
    const conv20 = this.buildConv_(conv19, '1', '29', true,
        {groups: 1632, padding: [1, 1, 1, 1]});
    const conv21 = this.buildConv_(conv20, '2', '29');

    const conv22 = this.buildConv_(conv21, '0', '', true);
    const pool1 = this.builder_.averagePool2d(await conv22);
    const reshape = this.builder_.reshape(pool1, [1, 1280]);
    return this.buildGemm_(reshape, '0');
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
