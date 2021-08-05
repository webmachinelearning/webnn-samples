'use strict';

import {buildConstantByNpy} from '../common/utils.js';

const autoPad = 'same-upper';
const strides = [2, 2];
const layout = 'nhwc';

// ResNet 101 V2 model with 'nhwc' layout
export class ResNet101V2Nhwc {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/resnet101v2_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: layout,
      labelUrl: './labels/labels1001.txt',
      inputDimensions: [1, 299, 299, 3],
    };
    this.outputDimensions = [1, 1001];
  }

  async buildConv_(input, nameIndices, options = {}, relu = true) {
    let prefix = this.weightsUrl_ + 'resnet_v2_101_';
    // Items in 'nameIndices' represent the indices of block, unit, conv
    // respectively, except two kinds of specific conv names:
    // 1. contains 'shortcut', e.g.
    // resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_weights.npy
    // 2. contains 'logits', e.g. resnet_v2_101_logits_weights.npy
    if (nameIndices[0] !== '' && nameIndices[1] !== '') {
      prefix += `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_`;
    }
    if (nameIndices[2] === 'shortcut') {
      prefix += 'shortcut';
    } else if (nameIndices[2] === 'logits') {
      prefix += nameIndices[2];
    } else {
      prefix += 'conv' + nameIndices[2];
    }
    const weightsName = prefix + '_weights.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    options.inputLayout = layout;
    options.filterLayout = 'ohwi';
    options.bias = bias;
    if (relu) {
      options.activation = this.builder_.relu();
    }
    return this.builder_.conv2d(input, weights, options);
  }

  async buildFusedBatchNorm_(input, nameIndices) {
    let prefix = this.weightsUrl_ + 'resnet_v2_101_';
    if (nameIndices[0] === 'postnorm') {
      prefix += 'postnorm';
    } else {
      prefix +=
          `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_preact`;
    }
    const mulParamName = prefix + '_FusedBatchNorm_mul_0_param.npy';
    const mulParam = await buildConstantByNpy(this.builder_, mulParamName);
    const addParamName = prefix + '_FusedBatchNorm_add_param.npy';
    const addParam = await buildConstantByNpy(this.builder_, addParamName);
    return this.builder_.relu(
        this.builder_.add(this.builder_.mul(input, mulParam), addParam));
  }

  async buildBottleneckV2_(
      input, nameIndices, downsample = false, shortcut = true) {
    let residual = input;

    const fusedBn = await this.buildFusedBatchNorm_(input, nameIndices);
    const conv1 = await this.buildConv_(
        fusedBn, nameIndices.concat(['1']), {autoPad});
    let conv2;
    if (downsample) {
      residual = await this.buildConv_(
          fusedBn, nameIndices.concat(['shortcut']), {autoPad}, false);
    }
    if (!downsample && shortcut) {
      residual = this.builder_.maxPool2d(
          input, {windowDimensions: [1, 1], strides, layout, autoPad});
      const padding = this.builder_.constant(
          {type: 'int32', dimensions: [4, 2]},
          new Int32Array([0, 0, 1, 1, 1, 1, 0, 0]));
      const pad = this.builder_.pad(conv1, padding);
      conv2 = await this.buildConv_(pad, nameIndices.concat(['2']), {strides});
    } else {
      conv2 = await this.buildConv_(
          conv1, nameIndices.concat(['2']), {autoPad});
    }
    const conv3 = await this.buildConv_(
        conv2, nameIndices.concat(['3']), {autoPad}, false);
    return this.builder_.add(conv3, residual);
  }

  async load(devicePreference) {
    const context = navigator.ml.createContext({devicePreference});
    this.builder_ = new MLGraphBuilder(context);
    const padding = this.builder_.constant(
        {type: 'int32', dimensions: [4, 2]},
        new Int32Array([0, 0, 3, 3, 3, 3, 0, 0]));

    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const pad = this.builder_.pad(input, padding);
    const conv1 = await this.buildConv_(pad, ['', '', '1'], {strides}, false);
    const pool = this.builder_.maxPool2d(
        conv1, {windowDimensions: [3, 3], strides, layout, autoPad});
    // Block 1
    const bottleneck1 = await this.buildBottleneckV2_(pool, ['1', '1'], true);
    const bottleneck2 = await this.buildBottleneckV2_(
        bottleneck1, ['1', '2'], false, false);
    const bottleneck3 = await this.buildBottleneckV2_(
        bottleneck2, ['1', '3']);

    // Block 2
    const bottleneck4 = await this.buildBottleneckV2_(
        bottleneck3, ['2', '1'], true);
    const bottleneck5 = await this.buildBottleneckV2_(
        bottleneck4, ['2', '2'], false, false);
    const bottleneck6 = await this.buildBottleneckV2_(
        bottleneck5, ['2', '3'], false, false);
    const bottleneck7 = await this.buildBottleneckV2_(
        bottleneck6, ['2', '4']);

    // Block 3
    const bottleneck8 = await this.buildBottleneckV2_(
        bottleneck7, ['3', '1'], true);
    const loop = async (node, num) => {
      if (num > 22) {
        return node;
      } else {
        const newNode = await this.buildBottleneckV2_(
            node, ['3', num.toString()], false, false);
        num++;
        return loop(newNode, num);
      }
    };
    const bottleneck9 = await loop(bottleneck8, 2);
    const bottleneck10 = await this.buildBottleneckV2_(
        bottleneck9, ['3', '23']);

    // Block 4
    const bottleneck11 = await this.buildBottleneckV2_(
        bottleneck10, ['4', '1'], true);
    const bottleneck12 = await this.buildBottleneckV2_(
        bottleneck11, ['4', '2'], false, false);
    const bottleneck13 = await this.buildBottleneckV2_(
        bottleneck12, ['4', '3'], false, false);

    const fusedBn =
        await this.buildFusedBatchNorm_(bottleneck13, ['postnorm']);
    const mean = this.builder_.reduceMean(
        fusedBn, {keepDimensions: true, axes: [1, 2]});
    const conv2 = await this.buildConv_(
        mean, ['', '', 'logits'], {autoPad}, false);
    const reshape = this.builder_.reshape(conv2, [1, -1]);
    return this.builder_.softmax(reshape);
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

  compute(inputBuffer, outputBuffer) {
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    this.graph_.compute(inputs, outputs);
  }
}
