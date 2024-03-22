'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

/* eslint max-len: ["error", {"code": 120}] */

// MobileNet V2 model with 'nhwc' input layout
export class MobileNetV2Nhwc {
  constructor() {
    this.context_ = null;
    this.deviceType_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/mobilenetv2_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: 'nhwc',
      labelUrl: './labels/labels1001.txt',
      inputDimensions: [1, 224, 224, 3],
    };
    this.outputDimensions = [1, 1001];
  }

  async buildConv_(input, weightsSubName, biasSubName, relu6, options) {
    const weightsName = this.weightsUrl_ + 'Const_' + weightsSubName + '.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = this.weightsUrl_ + 'MobilenetV2_' + biasSubName + '_bias.npy';
    const bias = buildConstantByNpy(this.builder_, biasName);
    options.inputLayout = 'nhwc';
    options.bias = await bias;
    // WebNN spec drops autoPad support, compute the explicit padding instead.
    if (options.autoPad == 'same-upper') {
      options.padding =
        computePadding2DForAutoPad(
            /* nwhc */[await input.shape()[1], await input.shape()[2]],
            /* ohwi or ihwo */[weights.shape()[1], weights.shape()[2]],
            options.strides, options.dilations, options.autoPad);
    }
    if (relu6) {
      // TODO: Set clamp activation to options once it's supported in
      // WebNN DML backend.
      // Implement `clip` by `clamp` of  WebNN API
      if (this.deviceType_ == 'gpu') {
        return this.builder_.clamp(
            this.builder_.conv2d(await input, weights, options),
            {minValue: 0, maxValue: 6});
      } else {
        options.activation = this.builder_.clamp({minValue: 0, maxValue: 6});
      }
    }
    return this.builder_.conv2d(input, weights, options);
  }

  async buildLinearBottleneck_(input, weightsNameArray, biasName, dwiseOptions, shortcut = true) {
    const autoPad = 'same-upper';
    const biasPrefix = 'expanded_conv_' + biasName;

    dwiseOptions.autoPad = autoPad;
    dwiseOptions.filterLayout = 'ihwo';

    const conv1x1Relu6 = this.buildConv_(
        await input,
        weightsNameArray[0],
        `${biasPrefix}_expand_Conv2D`,
        true,
        {autoPad, filterLayout: 'ohwi'},
    );
    const dwise3x3Relu6 = this.buildConv_(
        await conv1x1Relu6,
        weightsNameArray[1],
        `${biasPrefix}_depthwise_depthwise`,
        true,
        dwiseOptions,
    );
    const conv1x1Linear = this.buildConv_(
        await dwise3x3Relu6,
        weightsNameArray[2],
        `${biasPrefix}_project_Conv2D`,
        false,
        {autoPad, filterLayout: 'ohwi'},
    );

    if (shortcut) {
      return this.builder_.add(await input, await conv1x1Linear);
    }
    return await conv1x1Linear;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.deviceType_ = contextOptions.deviceType;
    this.builder_ = new MLGraphBuilder(this.context_);
    const strides = [2, 2];
    const autoPad = 'same-upper';
    const filterLayout = 'ohwi';
    const input = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });
    const conv0 = this.buildConv_(
        input, '90', 'Conv_Conv2D', true, {strides, autoPad, filterLayout});
    const conv1 = this.buildConv_(
        await conv0, '238', 'expanded_conv_depthwise_depthwise', true, {autoPad, groups: 32, filterLayout: 'ihwo'});
    const conv2 = this.buildConv_(
        await conv1, '167', 'expanded_conv_project_Conv2D', false, {autoPad, filterLayout});
    const bottleneck0 = this.buildLinearBottleneck_(
        await conv2, ['165', '99', '73'], '1', {strides, groups: 96}, false);
    const bottleneck1 = this.buildLinearBottleneck_(
        bottleneck0, ['3', '119', '115'], '2', {groups: 144});
    const bottleneck2 = this.buildLinearBottleneck_(
        bottleneck1, ['255', '216', '157'], '3', {strides, groups: 144}, false);
    const bottleneck3 = this.buildLinearBottleneck_(
        bottleneck2, ['227', '221', '193'], '4', {groups: 192});
    const bottleneck4 = this.buildLinearBottleneck_(
        bottleneck3, ['243', '102', '215'], '5', {groups: 192});
    const bottleneck5 = this.buildLinearBottleneck_(
        bottleneck4, ['226', '163', '229'], '6', {strides, groups: 192}, false);
    const bottleneck6 = this.buildLinearBottleneck_(
        bottleneck5, ['104', '254', '143'], '7', {groups: 384});
    const bottleneck7 = this.buildLinearBottleneck_(
        bottleneck6, ['25', '142', '202'], '8', {groups: 384});
    const bottleneck8 = this.buildLinearBottleneck_(
        bottleneck7, ['225', '129', '98'], '9', {groups: 384});
    const bottleneck9 = this.buildLinearBottleneck_(
        bottleneck8, ['169', '2', '246'], '10', {groups: 384}, false);
    const bottleneck10 = this.buildLinearBottleneck_(
        bottleneck9, ['162', '87', '106'], '11', {groups: 576});
    const bottleneck11 = this.buildLinearBottleneck_(
        bottleneck10, ['52', '22', '40'], '12', {groups: 576});
    const bottleneck12 = this.buildLinearBottleneck_(
        bottleneck11, ['114', '65', '242'], '13', {strides, groups: 576}, false);
    const bottleneck13 = this.buildLinearBottleneck_(
        bottleneck12, ['203', '250', '92'], '14', {groups: 960});
    const bottleneck14 = this.buildLinearBottleneck_(
        bottleneck13, ['133', '130', '258'], '15', {groups: 960});
    const bottleneck15 = this.buildLinearBottleneck_(
        bottleneck14, ['60', '248', '100'], '16', {groups: 960}, false);
    const conv3 = this.buildConv_(
        await bottleneck15, '71', 'Conv_1_Conv2D', true, {autoPad, filterLayout});

    const averagePool2d = this.builder_.averagePool2d(await conv3, {windowDimensions: [7, 7], layout: 'nhwc'});
    const conv4 = this.buildConv_(
        averagePool2d, '222', 'Logits_Conv2d_1c_1x1_Conv2D', false, {autoPad, filterLayout});
    const reshape = this.builder_.reshape(await conv4, [1, 1001]);
    return await this.builder_.softmax(reshape);
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
