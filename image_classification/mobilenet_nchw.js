'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// MobileNet V2 model with 'nchw' input layout
export class MobileNetV2Nchw {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
    };
  }

  async buildConv_(input, name, shouldRelu6 = true, options = undefined) {
    const prefix = './weights/mobilenet_nchw/conv_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights =
        await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias =
        await buildConstantByNpy(this.builder_, biasName);
    const conv = this.builder_.add(
        this.builder_.conv2d(input, weights, options),
        this.builder_.reshape(bias, [1, -1, 1, 1]));
    if (shouldRelu6) {
      return this.builder_.clamp(
          conv,
          {
            minValue: this.builder_.constant(0.),
            maxValue: this.builder_.constant(6.0),
          });
    } else {
      return conv;
    }
  }

  async buildGemm_(input, name) {
    const prefix = './weights/mobilenet_nchw/gemm_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const options = {c: bias, bTranspose: true};
    return this.builder_.gemm(input, weights, options);
  }

  async buildBottleneck_(
      input, convNameArray, groups, strides = false, shouldAdd = true) {
    const conv1x1 = await this.buildConv_(input, convNameArray[0]);
    const options = {
      padding: [1, 1, 1, 1],
      groups: groups,
    };
    if (strides) {
      options.strides = [2, 2];
    }
    const conv3x3 = await this.buildConv_(
        conv1x1, convNameArray[1], true, options);
    const conv1x1NotClip = await this.buildConv_(
        conv3x3, convNameArray[2], false);
    if (shouldAdd) {
      return this.builder_.add(input, conv1x1NotClip);
    } else {
      return conv1x1NotClip;
    }
  }

  async buildBottleneckMore_(
      input, convNameArray, groupsArrary, strides = true) {
    const out1 = await this.buildBottleneck_(
        input, convNameArray.slice(0, 3), groupsArrary[0], strides, false);
    const out2 = await this.buildBottleneck_(
        out1, convNameArray.slice(3, 6), groupsArrary[1]);
    if (convNameArray.length >= 9) {
      const out3 = await this.buildBottleneck_(
          out2, convNameArray.slice(6, 9), groupsArrary[1]);
      if (convNameArray.length === 12) {
        return await this.buildBottleneck_(
            out3, convNameArray.slice(9, 12), groupsArrary[1]);
      } else {
        return out3;
      }
    } else {
      return out2;
    }
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    const data = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const conv0 = await this.buildConv_(
        data, 0, true, {padding: [1, 1, 1, 1], strides: [2, 2]});
    const conv2 = await this.buildConv_(
        conv0, 2, true, {padding: [1, 1, 1, 1], groups: 32});
    const conv4 = await this.buildConv_(conv2, 4, false);
    const add15 = await this.buildBottleneckMore_(
        conv4, [5, 7, 9, 10, 12, 14], [96, 144]);
    const add32 = await this.buildBottleneckMore_(
        add15, [16, 18, 20, 21, 23, 25, 27, 29, 31], [144, 192]);
    const add55 = await this.buildBottleneckMore_(
        add32, [33, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54], [192, 384]);
    const add72 = await this.buildBottleneckMore_(
        add55, [56, 58, 60, 61, 63, 65, 67, 69, 71], [384, 576], false);
    const add89 = await this.buildBottleneckMore_(
        add72, [73, 75, 77, 78, 80, 82, 84, 86, 88], [576, 960]);
    const conv94 = await this.buildBottleneck_(
        add89, [90, 92, 94], 960, false, false);
    const conv95 = await this.buildConv_(conv94, 95, true);
    const pool97 = this.builder_.averagePool2d(conv95);
    const reshape103 = this.builder_.reshape(pool97, [1, -1]);
    const gemm104 = await this.buildGemm_(reshape103, 104);
    return this.builder_.softmax(gemm104);
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
