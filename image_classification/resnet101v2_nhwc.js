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
    this.weightsUrl_ = 'https://webmachinelearning.github.io/test-data/' +
        'models/resnet101v2_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: layout,
      labelUrl: './labels/labels1001.txt',
      inputDimensions: [1, 299, 299, 3],
    };
  }

  async buildConv_(input, nameArray, options = undefined, relu = true) {
    let prefix = this.weightsUrl_ + 'resnet_v2_101_';
    // Items in 'nameArray' represent the indexes of block, unit, conv
    // respectively, except two kinds of specific conv names:
    // 1. contains 'shortcut', e.g.
    // resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_weights.npy
    // 2. contains 'logits', e.g. resnet_v2_101_logits_weights.npy
    if (nameArray[0] !== '' && nameArray[1] !== '') {
      prefix += `block${nameArray[0]}_unit_${nameArray[1]}_bottleneck_v2_`;
    }
    if (nameArray[2] === 'shortcut') {
      prefix += 'shortcut';
    } else if (nameArray[2] === 'logits') {
      prefix += nameArray[2];
    } else {
      prefix += 'conv' + nameArray[2];
    }
    const weightsName = prefix + '_weights.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    if (options !== undefined) {
      options.inputLayout = layout;
      options.filterLayout = 'ohwi';
    } else {
      options = {inputLayout: layout, filterLayout: 'ohwi'};
    }
    const add = this.builder_.add(
        this.builder_.conv2d(input, weights, options),
        this.builder_.reshape(bias, [1, 1, 1, -1]));
    if (relu) {
      return this.builder_.relu(add);
    }
    return add;
  }

  async buildFusedBatchNorm_(input, nameArray) {
    let prefix = this.weightsUrl_ + 'resnet_v2_101_';
    if (nameArray[0] === 'postnorm') {
      prefix += 'postnorm';
    } else {
      prefix +=
          `block${nameArray[0]}_unit_${nameArray[1]}_bottleneck_v2_preact`;
    }
    const mulParamName = prefix + '_FusedBatchNorm_mul_0_param.npy';
    const mulParam = await buildConstantByNpy(this.builder_, mulParamName);
    const addParamName = prefix + '_FusedBatchNorm_add_param.npy';
    const addParam = await buildConstantByNpy(this.builder_, addParamName);
    return this.builder_.relu(
        this.builder_.add(this.builder_.mul(input, mulParam), addParam));
  }

  async buildLinearBottleneck1_(input, nameArray) {
    const fusedBatchNorm =
        await this.buildFusedBatchNorm_(input, nameArray);

    const conv0 = await this.buildConv_(
        fusedBatchNorm, nameArray.concat(['1']), {autoPad});
    const conv3x3 = await this.buildConv_(
        conv0, nameArray.concat(['2']), {autoPad});
    const conv1 = await this.buildConv_(
        conv3x3, nameArray.concat(['3']), {autoPad}, false);
    const conv2 = await this.buildConv_(
        fusedBatchNorm, nameArray.concat(['shortcut']), {autoPad}, false);

    return this.builder_.add(conv1, conv2);
  }

  async buildLinearBottleneck2_(input, nameArray, shortcut = true) {
    const fusedBatchNorm =
        await this.buildFusedBatchNorm_(input, nameArray);

    const conv0 = await this.buildConv_(
        fusedBatchNorm, nameArray.concat(['1']), {autoPad});
    let conv3x3;
    if (shortcut) {
      const padding = this.builder_.constant(
          {type: 'int32', dimensions: [4, 2]},
          new Int32Array([0, 0, 1, 1, 1, 1, 0, 0]));
      const pad = this.builder_.pad(conv0, padding);
      conv3x3 = await this.buildConv_(
          pad, nameArray.concat(['2']), {strides});
    } else {
      conv3x3 = await this.buildConv_(
          conv0, nameArray.concat(['2']), {autoPad});
    }
    const conv2 = await this.buildConv_(
        conv3x3, nameArray.concat(['3']), {autoPad}, false);

    if (shortcut) {
      const pool = this.builder_.maxPool2d(
          input, {windowDimensions: [1, 1], strides, layout, autoPad});
      return this.builder_.add(pool, conv2);
    }
    return this.builder_.add(input, conv2);
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    const padding = this.builder_.constant(
        {type: 'int32', dimensions: [4, 2]},
        new Int32Array([0, 0, 3, 3, 3, 3, 0, 0]));

    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const pad = this.builder_.pad(input, padding);
    const conv0 = await this.buildConv_(pad, ['', '', '1'], {strides}, false);
    const pool = this.builder_.maxPool2d(
        conv0, {windowDimensions: [3, 3], strides, layout, autoPad});
    // Block 1
    const bottleneck0 = await this.buildLinearBottleneck1_(pool, ['1', '1']);
    const bottleneck1 = await this.buildLinearBottleneck2_(
        bottleneck0, ['1', '2'], false);
    const bottleneck2 = await this.buildLinearBottleneck2_(
        bottleneck1, ['1', '3']);

    // Block 2
    const bottleneck3 = await this.buildLinearBottleneck1_(
        bottleneck2, ['2', '1']);
    const bottleneck4 = await this.buildLinearBottleneck2_(
        bottleneck3, ['2', '2'], false);
    const bottleneck5 = await this.buildLinearBottleneck2_(
        bottleneck4, ['2', '3'], false);
    const bottleneck6 = await this.buildLinearBottleneck2_(
        bottleneck5, ['2', '4']);

    // Block 3
    const bottleneck7 = await this.buildLinearBottleneck1_(
        bottleneck6, ['3', '1']);
    const loop = async (node, num) => {
      if (num > 22) {
        return node;
      } else {
        const newNode = await this.buildLinearBottleneck2_(
            node, ['3', num.toString()], false);
        num++;
        return loop(newNode, num);
      }
    };
    const bottleneck8 = await loop(bottleneck7, 2);
    const bottleneck9 = await this.buildLinearBottleneck2_(
        bottleneck8, ['3', '23']);

    // Block 4
    const bottleneck10 = await this.buildLinearBottleneck1_(
        bottleneck9, ['4', '1']);
    const bottleneck11 = await this.buildLinearBottleneck2_(
        bottleneck10, ['4', '2'], false);
    const bottleneck12 = await this.buildLinearBottleneck2_(
        bottleneck11, ['4', '3'], false);

    const fusedBatchNorm =
        await this.buildFusedBatchNorm_(bottleneck12, ['postnorm']);
    const mean = this.builder_.reduceMean(
        fusedBatchNorm, {keepDimensions: true, axes: [1, 2]});
    const conv1 = await this.buildConv_(
        mean, ['', '', 'logits'], {autoPad}, false);
    const reshape = this.builder_.reshape(conv1, [1, -1]);
    return this.builder_.softmax(reshape);
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
