'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

// SSD MobileNet V2 Face model with 'nhwc' layout.
export class SsdMobilenetV2FaceNhwc {
  constructor() {
    this.context_ = null;
    this.deviceType_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/ssd_mobilenetv2_face_nhwc/weights/';
    this.inputOptions = {
      inputLayout: 'nhwc',
      margin: [1.2, 1.2, 0.8, 1.1],
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      boxSize: 4,
      numClasses: 2,
      numBoxes: [1083, 600, 150, 54, 24, 6],
      inputDimensions: [1, 300, 300, 3],
    };
    this.outputsInfo = {
      'biasAdd0': [1, 19, 19, 12],
      'biasAdd3': [1, 19, 19, 6],
      'biasAdd6': [1, 10, 10, 24],
      'biasAdd9': [1, 10, 10, 12],
      'biasAdd12': [1, 5, 5, 24],
      'biasAdd15': [1, 5, 5, 12],
      'biasAdd18': [1, 3, 3, 24],
      'biasAdd21': [1, 3, 3, 12],
      'biasAdd24': [1, 2, 2, 24],
      'biasAdd27': [1, 2, 2, 12],
      'biasAdd30': [1, 1, 1, 24],
      'biasAdd33': [1, 1, 1, 12],
    };
  }

  async buildConv_(input, nameArray, relu6 = true, options = undefined) {
    // nameArray: 0: keyword, 1: indice or suffix
    let prefix = this.weightsUrl_;
    const weightSuffix = '_weights.npy';
    let biasSuffix = '_Conv2D_bias.npy';

    if (nameArray[0].includes('expanded')) {
      prefix += 'FeatureExtractor_MobilenetV2_expanded_conv_';
      if (nameArray[0].includes('depthwise')) {
        prefix += nameArray[1] === '0' ?
            'depthwise_depthwise' : `${nameArray[1]}_depthwise_depthwise`;
        biasSuffix = '_bias.npy';
      } else if (nameArray[0].includes('project')) {
        prefix += nameArray[1] === '0' ? 'project' : `${nameArray[1]}_project`;
      } else {
        prefix += `${nameArray[1]}_expand`;
      }
    } else if (nameArray[0] === 'Class' || nameArray[0] === 'BoxEncoding') {
      prefix += `BoxPredictor_${nameArray[1]}_${nameArray[0]}Predictor`;
    } else if (nameArray[0].includes('layer')) { // layer_19_1 or layer_19_2
      prefix += `FeatureExtractor_MobilenetV2_${nameArray[0]}_Conv2d_\
${nameArray[1]}`;
    } else {
      prefix += `${nameArray[0]}`;
    }

    const weightsName = prefix + weightSuffix;
    const weights = buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + biasSuffix;
    const bias = buildConstantByNpy(this.builder_, biasName);
    if (options !== undefined) {
      options.inputLayout = 'nhwc';
      options.filterLayout = 'ohwi';
    } else {
      options = {
        inputLayout: 'nhwc',
        filterLayout: 'ohwi',
      };
    }
    if (nameArray[0].includes('depthwise')) {
      options.filterLayout = 'ihwo';
    }
    options.bias = await bias;
    const inputShape = (await input).shape();
    const weightsShape = (await weights).shape();
    options.padding = computePadding2DForAutoPad(
        /* nhwc */[inputShape[1], inputShape[2]],
        /* ohwi or ihwo */[weightsShape[1], weightsShape[2]],
        options.strides, options.dilations, 'same-upper');
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

  async buildLinearBottleneck_(
      input, indice, shortcut = true, groups, stridesNode) {
    let convOptions;
    const dwiseOptions = {groups};
    const strides = [2, 2];
    if (stridesNode === 'convRelu6') {
      convOptions = {strides};
    }
    if (stridesNode === 'dwiseRelu6') {
      dwiseOptions.strides = strides;
    }
    const convRelu6Keyword = indice === '0' ?
          'FeatureExtractor_MobilenetV2_Conv' : 'expanded';

    const convRelu6 = this.buildConv_(
        input, [convRelu6Keyword, indice], true, convOptions);
    const dwiseRelu6 = this.buildConv_(
        convRelu6, ['expanded_depthwise', indice], true, dwiseOptions);
    const convLinear = this.buildConv_(
        dwiseRelu6, ['expanded_project', indice], false);

    if (shortcut) {
      return this.builder_.add(await input, await convLinear);
    }
    return await convLinear;
  }


  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.deviceType_ = contextOptions.deviceType;
    this.builder_ = new MLGraphBuilder(this.context_);
    const input = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });

    const bottleneck0 = this.buildLinearBottleneck_(
        input, '0', false, 32, 'convRelu6');
    const bottleneck1 = this.buildLinearBottleneck_(
        bottleneck0, '1', false, 96, 'dwiseRelu6');
    const bottleneck2 = this.buildLinearBottleneck_(
        bottleneck1, '2', true, 144);
    const bottleneck3 = this.buildLinearBottleneck_(
        bottleneck2, '3', false, 144, 'dwiseRelu6');
    const bottleneck4 = this.buildLinearBottleneck_(
        bottleneck3, '4', true, 192);
    const bottleneck5 = this.buildLinearBottleneck_(
        bottleneck4, '5', true, 192);
    const bottleneck6 = this.buildLinearBottleneck_(
        bottleneck5, '6', false, 192, 'dwiseRelu6');
    const bottleneck7 = this.buildLinearBottleneck_(
        bottleneck6, '7', true, 384);
    const bottleneck8 = this.buildLinearBottleneck_(
        bottleneck7, '8', true, 384);
    const bottleneck9 = this.buildLinearBottleneck_(
        bottleneck8, '9', true, 384);
    const bottleneck10 = this.buildLinearBottleneck_(
        bottleneck9, '10', false, 384);
    const bottleneck11 = this.buildLinearBottleneck_(
        bottleneck10, '11', true, 576);
    const bottleneck12 = this.buildLinearBottleneck_(
        bottleneck11, '12', true, 576);
    const conv13Relu6 = this.buildConv_(
        bottleneck12, ['expanded', '13']);
    const dwise13Relu6 = this.buildConv_(
        conv13Relu6,
        ['expanded_depthwise', '13'],
        true,
        {groups: 576, strides: [2, 2]});
    const convLinear13 = this.buildConv_(
        dwise13Relu6, ['expanded_project', '13'], false);

    const biasAdd0 = this.buildConv_(
        conv13Relu6, ['BoxEncoding', '0'], false);
    const biasAdd3 = this.buildConv_(
        conv13Relu6, ['Class', '0'], false);

    const bottleneck14 = this.buildLinearBottleneck_(
        convLinear13, '14', true, 960);
    const bottleneck15 = this.buildLinearBottleneck_(
        bottleneck14, '15', true, 960);
    const bottleneck16 = this.buildLinearBottleneck_(
        bottleneck15, '16', false, 960);

    const conv17Relu6 = this.buildConv_(
        bottleneck16, ['FeatureExtractor_MobilenetV2_Conv_1']);
    const biasAdd6 = this.buildConv_(
        conv17Relu6, ['BoxEncoding', '1'], false);
    const biasAdd9 = this.buildConv_(
        conv17Relu6, ['Class', '1'], false);

    const conv18Relu6 = this.buildConv_(
        conv17Relu6, ['layer_19_1', '2_1x1_256']);
    const conv19Relu6 = this.buildConv_(
        conv18Relu6, ['layer_19_2', '2_3x3_s2_512'], true, {strides: [2, 2]});
    const biasAdd12 = this.buildConv_(
        conv19Relu6, ['BoxEncoding', '2'], false);
    const biasAdd15 = this.buildConv_(
        conv19Relu6, ['Class', '2'], false);

    const conv20Relu6 = this.buildConv_(
        conv19Relu6, ['layer_19_1', '3_1x1_128']);
    const conv21Relu6 = this.buildConv_(
        conv20Relu6, ['layer_19_2', '3_3x3_s2_256'], true, {strides: [2, 2]});
    const biasAdd18 = this.buildConv_(
        conv21Relu6, ['BoxEncoding', '3'], false);
    const biasAdd21 = this.buildConv_(
        conv21Relu6, ['Class', '3'], false);

    const conv22Relu6 = this.buildConv_(
        conv21Relu6, ['layer_19_1', '4_1x1_128']);
    const conv23Relu6 = this.buildConv_(
        conv22Relu6, ['layer_19_2', '4_3x3_s2_256'], true, {strides: [2, 2]});
    const biasAdd24 = this.buildConv_(
        conv23Relu6, ['BoxEncoding', '4'], false);
    const biasAdd27 = this.buildConv_(
        conv23Relu6, ['Class', '4'], false);

    const conv24Relu6 = this.buildConv_(
        conv23Relu6, ['layer_19_1', '5_1x1_64']);
    const conv25Relu6 = this.buildConv_(
        conv24Relu6, ['layer_19_2', '5_3x3_s2_128'], true, {strides: [2, 2]});
    const biasAdd30 = this.buildConv_(
        conv25Relu6, ['BoxEncoding', '5'], false);
    const biasAdd33 = this.buildConv_(
        conv25Relu6, ['Class', '5'], false);

    return {
      biasAdd0: await biasAdd0,
      biasAdd3: await biasAdd3,
      biasAdd6: await biasAdd6,
      biasAdd9: await biasAdd9,
      biasAdd12: await biasAdd12,
      biasAdd15: await biasAdd15,
      biasAdd18: await biasAdd18,
      biasAdd21: await biasAdd21,
      biasAdd24: await biasAdd24,
      biasAdd27: await biasAdd27,
      biasAdd30: await biasAdd30,
      biasAdd33: await biasAdd33,
    };
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build(outputOperand);
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  async compute(inputBuffer, outputs) {
    const inputs = {'input': inputBuffer};
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results;
  }
}
