'use strict';

import {buildConstantByNpy} from '../common/utils.js';

/* eslint max-len: ["error", {"code": 120}] */

// MobileNet V2 baseline model with nhwc layout
export class MobileNetNhwc {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
  }

  async buildConv_(input, weightsSubName, biasSubName, shouldRelu6, options = undefined) {
    const prefix = './weights/mobilenet_nhwc/';
    const weightsName = prefix + 'Const_' + weightsSubName + '.npy';
    let weights = await buildConstantByNpy(this.builder_, weightsName);
    if (biasSubName.includes('depthwise')) {
      // DepthwiseConv2D's filterLayout is 'ihwo', should transpose to 'hwio'
      weights = await buildConstantByNpy(this.builder_, weightsName, true);
    }
    const biasName = prefix + 'MobilenetV2_' + biasSubName + '_bias.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    if (options !== undefined) {
      options.inputLayout = 'nhwc';
    } else {
      options = {inputLayout: 'nhwc'};
    }
    const add = this.builder_.add(
        this.builder_.conv2d(input, weights, options),
        this.builder_.reshape(bias, [1, 1, 1, -1]));
    // `relu6` in TFLite equals to `clamp` in WebNN API
    if (shouldRelu6) {
      return this.builder_.clamp(
          add,
          {
            minValue: this.builder_.constant(0.),
            maxValue: this.builder_.constant(6.0),
          });
    }
    return add;
  }

  async buildFire_(input, weightsNameIndexes, biasNameIndex, depthwiseOptions = undefined) {
    const autoPad = 'same-lower';
    if (depthwiseOptions !== undefined) {
      depthwiseOptions.autoPad = autoPad;
      depthwiseOptions.filterLayout = 'hwio';
    } else {
      depthwiseOptions = {autoPad, filterLayout: 'hwio'};
    }
    const biasPrefix = 'expanded_conv_' + biasNameIndex;
    const covOptions = {autoPad, filterLayout: 'ohwi'};
    const conv0 = await this.buildConv_(
        input, weightsNameIndexes[0], `${biasPrefix}_expand_Conv2D`, true, covOptions);
    const conv1 = await this.buildConv_(
        conv0, weightsNameIndexes[1], `${biasPrefix}_depthwise_depthwise`, true, depthwiseOptions);
    return await this.buildConv_(
        conv1, weightsNameIndexes[2], `${biasPrefix}_project_Conv2D`, false, covOptions);
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    const strides = [2, 2];
    const autoPad = 'same-lower';
    const filterLayout = 'ohwi';
    const input = this.builder_.input('input', {type: 'float32', dimensions: [1, 224, 224, 3]});
    const conv0 = await this.buildConv_(
        input, '90', 'Conv_Conv2D', true, {strides, autoPad, filterLayout});
    const conv1 = await this.buildConv_(
        conv0, '238', 'expanded_conv_depthwise_depthwise', true, {autoPad, groups: 32, filterLayout: 'hwio'});
    const conv2 = await this.buildConv_(
        conv1, '167', 'expanded_conv_project_Conv2D', false, {autoPad, filterLayout});
    const fire0 = await this.buildFire_(conv2, ['165', '99', '73'], '1', {strides, groups: 96});
    const fire1 = await this.buildFire_(fire0, ['3', '119', '115'], '2', {groups: 144});
    const add0 = this.builder_.add(fire0, fire1);
    const fire2 = await this.buildFire_(add0, ['255', '216', '157'], '3', {strides, groups: 144});
    const fire3 = await this.buildFire_(fire2, ['227', '221', '193'], '4', {groups: 192});
    const add1 = this.builder_.add(fire2, fire3);
    const fire4 = await this.buildFire_(add1, ['243', '102', '215'], '5', {groups: 192});
    const add2 = this.builder_.add(add1, fire4);
    const fire5 = await this.buildFire_(add2, ['226', '163', '229'], '6', {strides, groups: 192});
    const fire6 = await this.buildFire_(fire5, ['104', '254', '143'], '7', {groups: 384});
    const add3 = this.builder_.add(fire5, fire6);
    const fire7 = await this.buildFire_(add3, ['25', '142', '202'], '8', {groups: 384});
    const add4 = this.builder_.add(add3, fire7);
    const fire8 = await this.buildFire_(add4, ['225', '129', '98'], '9', {groups: 384});
    const add5 = this.builder_.add(add4, fire8);
    const fire9 = await this.buildFire_(add5, ['169', '2', '246'], '10', {groups: 384});
    const fire10 = await this.buildFire_(fire9, ['162', '87', '106'], '11', {groups: 576});
    const add6 = this.builder_.add(fire9, fire10);
    const fire11 = await this.buildFire_(add6, ['52', '22', '40'], '12', {groups: 576});
    const add7 = this.builder_.add(add6, fire11);
    const fire12 = await this.buildFire_(add7, ['114', '65', '242'], '13', {strides, groups: 576});
    const fire13 = await this.buildFire_(fire12, ['203', '250', '92'], '14', {groups: 960});
    const add8 = this.builder_.add(fire12, fire13);
    const fire14 = await this.buildFire_(add8, ['133', '130', '258'], '15', {groups: 960});
    const add9 = this.builder_.add(add8, fire14);
    const fire15 = await this.buildFire_(add9, ['60', '248', '100'], '16', {groups: 960});
    const conv3 = await this.buildConv_(fire15, '71', 'Conv_1_Conv2D', true, {autoPad, filterLayout});

    const averagePool2d = this.builder_.averagePool2d(conv3, {windowDimensions: [7, 7], layout: 'nhwc'});
    const conv4 = await this.buildConv_(
        averagePool2d, '222', 'Logits_Conv2d_1c_1x1_Conv2D', false, {autoPad, filterLayout});
    const reshape = this.builder_.reshape(conv4, [1, -1]);
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
