'use strict';

import {buildConstantByNpy, weightsOrigin} from '../common/utils.js';

/* eslint max-len: ["error", {"code": 120}] */

// DeepLab V3 MobileNet V2 model with 'nchw' input layout
export class DeepLabV3MNV2Nchw {
  constructor() {
    this.context_ = null;
    this.deviceType_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/deeplabv3_mnv2_nchw/weights/';
    // Shares the same bias files with 'nhwc' layout
    this.biasUrl_ = weightsOrigin() +
      '/test-data/models/deeplabv3_mnv2_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      scaledFlag: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels.txt',
      inputDimensions: [1, 3, 513, 513],
    };
    this.outputDimensions = [1, 21, 513, 513];
  }

  async buildConv_(input, nameArray, activation = 'relu6', options = {}) {
    // nameArray: 0: bias name prefix, 1: depthWise Conv2D's bias name suffix, 2: indice of weight name
    const biasPrefix = this.biasUrl_ + nameArray[0];
    const weightsName = `${this.weightsUrl_}const_fold_opt__${nameArray[2]}.npy`;
    let biasName = biasPrefix + '_bn_offset.npy';
    if (nameArray[0].includes('depthwise')) {
      biasName = `${biasPrefix}_bn_offset.npy`;
      if (nameArray[1] !== '') {
        biasName = `${biasPrefix}_${nameArray[1]}.npy`;
      }
    } else if (nameArray[0] === 'logits_semantic') {
      biasName = biasPrefix + '_biases.npy';
    }

    const weights = buildConstantByNpy(this.builder_, weightsName);
    const bias = buildConstantByNpy(this.builder_, biasName);

    options.bias = await bias;
    if (activation === 'relu6') {
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
    } else if (activation === 'relu') {
      options.activation = this.builder_.relu();
    } else {
      options.activation = undefined;
    }
    return this.builder_.conv2d(await input, await weights, options);
  }

  async buildLinearBottleneck_(input, nameArray, dwiseOptions, shortcut = true) {
    // nameArray: 0: indice of bias name, 1: indice of conv1x1Relu6's weight name,
    // 2: indice of dwise3x3Relu6's weight name, 3: indice of conv1x1Linear's weight name
    const biasPrefix = 'MobilenetV2_expanded_conv_' + nameArray[0];
    let dwBiasSuffix = 'depthwise_bn_offset';
    if (Number.parseInt(nameArray[0]) > 6) {
      dwBiasSuffix = 'BatchNorm_FusedBatchNorm';
    }
    const conv1x1Relu6 = this.buildConv_(
        input,
        [`${biasPrefix}_expand_Conv2D`, dwBiasSuffix, nameArray[1]]);
    const dwise3x3Relu6 = this.buildConv_(
        conv1x1Relu6,
        [`${biasPrefix}_depthwise`, dwBiasSuffix, nameArray[2]],
        'relu6',
        dwiseOptions);
    const conv1x1Linear = this.buildConv_(
        dwise3x3Relu6,
        [`${biasPrefix}_project_Conv2D`, dwBiasSuffix, nameArray[3]],
        'none');

    if (shortcut) {
      return this.builder_.add(await input, await conv1x1Linear);
    }
    return conv1x1Linear;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.deviceType_ = contextOptions.deviceType;
    this.builder_ = new MLGraphBuilder(this.context_);
    const strides = [2, 2];

    const input = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });
    const conv0 = this.buildConv_(
        input, ['MobilenetV2_Conv_Conv2D', '', '551'], 'relu6', {strides, padding: [1, 1, 1, 1]});
    const conv1 = this.buildConv_(
        conv0, ['MobilenetV2_expanded_conv_depthwise_depthwise', '', '543'], 'relu6',
        {padding: [1, 1, 1, 1], groups: 32});
    const conv2 = this.buildConv_(
        conv1, ['MobilenetV2_expanded_conv_project_Conv2D', '', '511'], 'none');
    const bottleneck0 = this.buildLinearBottleneck_(
        conv2, ['1', '537', '494', '534'], {strides, padding: [1, 1, 1, 1], groups: 96}, false);
    const bottleneck1 = this.buildLinearBottleneck_(
        bottleneck0, ['2', '447', '555', '523'], {padding: [1, 1, 1, 1], groups: 144});
    const bottleneck2 = this.buildLinearBottleneck_(
        bottleneck1, ['3', '520', '562', '542'], {strides, padding: [1, 1, 1, 1], groups: 144}, false);
    const bottleneck3 = this.buildLinearBottleneck_(
        bottleneck2, ['4', '503', '505', '489'], {padding: [1, 1, 1, 1], groups: 192});
    const bottleneck4 = this.buildLinearBottleneck_(
        bottleneck3, ['5', '446', '530', '522'], {padding: [1, 1, 1, 1], groups: 192});
    const bottleneck5 = this.buildLinearBottleneck_(
        bottleneck4, ['6', '491', '561', '538'], {padding: [1, 1, 1, 1], groups: 192}, false);
    const bottleneck6 = this.buildLinearBottleneck_(
        bottleneck5, ['7', '487', '560', '478'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]});
    const bottleneck7 = this.buildLinearBottleneck_(
        bottleneck6, ['8', '467', '536', '455'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]});
    const bottleneck8 = this.buildLinearBottleneck_(
        bottleneck7, ['9', '474', '524', '558'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]});
    const bottleneck9 = this.buildLinearBottleneck_(
        bottleneck8, ['10', '465', '556', '462'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]}, false);
    const bottleneck10 = this.buildLinearBottleneck_(
        bottleneck9, ['11', '453', '532', '450'], {padding: [2, 2, 2, 2], groups: 576, dilations: [2, 2]});
    const bottleneck11 = this.buildLinearBottleneck_(
        bottleneck10, ['12', '441', '554', '517'], {padding: [2, 2, 2, 2], groups: 576, dilations: [2, 2]});
    const bottleneck12 = this.buildLinearBottleneck_(
        bottleneck11, ['13', '544', '509', '479'], {padding: [2, 2, 2, 2], groups: 576, dilations: [2, 2]}, false);
    const bottleneck13 = this.buildLinearBottleneck_(
        bottleneck12, ['14', '482', '552', '512'], {padding: [4, 4, 4, 4], groups: 960, dilations: [4, 4]});
    const bottleneck14 = this.buildLinearBottleneck_(
        bottleneck13, ['15', '475', '495', '563'], {padding: [4, 4, 4, 4], groups: 960, dilations: [4, 4]});
    const bottleneck15 = this.buildLinearBottleneck_(
        bottleneck14, ['16', '500', '459', '539'], {padding: [4, 4, 4, 4], groups: 960, dilations: [4, 4]}, false);

    const conv3 = this.buildConv_(bottleneck15, ['aspp0_Conv2D', '', '553'], 'relu');
    const averagePool2d = this.builder_.averagePool2d(
        await bottleneck15, {windowDimensions: [65, 65], layout: 'nchw'});
    const conv4 = this.buildConv_(averagePool2d, ['image_pooling_Conv2D', '', '546'], 'relu');
    const resample0 = this.builder_.resample2d(
        await conv4, {sizes: [65, 65], mode: 'linear'});
    const concat = this.builder_.concat([await resample0, await conv3], 1);

    const conv5 = this.buildConv_(concat, ['concat_projection_Conv2D', '', '502'], 'relu');
    const conv6 = this.buildConv_(conv5, ['logits_semantic', '', '541'], 'none');
    const resample1 = this.builder_.resample2d(
        await conv6, {sizes: [65, 65], mode: 'linear'});
    return this.builder_.resample2d(
        resample1, {sizes: [513, 513], mode: 'linear'});
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
