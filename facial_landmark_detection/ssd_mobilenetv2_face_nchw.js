'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// SSD MobileNet V2 Face model with 'nchw' layout.
export class SsdMobilenetV2FaceNchw {
  constructor() {
    this.context_ = null;
    this.devicePreference_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/ssd_mobilenetv2_face_nchw/weights/';
    this.inputOptions = {
      inputLayout: 'nchw',
      margin: [1.2, 1.2, 0.8, 1.1],
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      boxSize: 4,
      numClasses: 2,
      numBoxes: [1083, 600, 150, 54, 24, 6],
      inputDimensions: [1, 3, 300, 300],
    };
    this.outputsInfo = {
      'biasAdd0': [1, 12, 19, 19],
      'biasAdd3': [1, 6, 19, 19],
      'biasAdd6': [1, 24, 10, 10],
      'biasAdd9': [1, 12, 10, 10],
      'biasAdd12': [1, 24, 5, 5],
      'biasAdd15': [1, 12, 5, 5],
      'biasAdd18': [1, 24, 3, 3],
      'biasAdd21': [1, 12, 3, 3],
      'biasAdd24': [1, 24, 2, 2],
      'biasAdd27': [1, 12, 2, 2],
      'biasAdd30': [1, 24, 1, 1],
      'biasAdd33': [1, 12, 1, 1],
    };
  }

  async buildConv_(input, nameArray, clip = true, options = undefined) {
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
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + biasSuffix;
    const bias = await buildConstantByNpy(this.builder_, biasName);
    if (options !== undefined) {
      options.autoPad = 'same-upper';
    } else {
      options = {
        autoPad: 'same-upper',
      };
    }
    options.bias = bias;
    if (clip) {
      // TODO: Set clamp activation to options once it's supported in
      // WebNN DML backend.
      // Implement `clip` by `clamp` of  WebNN API
      if (this.devicePreference_ == 'gpu') {
        return this.builder_.clamp(
            this.builder_.conv2d(input, weights, options),
            {minValue: 0, maxValue: 6});
      } else {
        options.activation = this.builder_.clamp({minValue: 0, maxValue: 6});
      }
    }
    return this.builder_.conv2d(input, weights, options);
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

    const convRelu6 = await this.buildConv_(
        input, [convRelu6Keyword, indice], true, convOptions);
    const dwiseRelu6 = await this.buildConv_(
        convRelu6, ['expanded_depthwise', indice], true, dwiseOptions);
    const convLinear = await this.buildConv_(
        dwiseRelu6, ['expanded_project', indice], false);

    if (shortcut) {
      return this.builder_.add(input, convLinear);
    }
    return convLinear;
  }


  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.devicePreference_ = contextOptions.devicePreference;
    this.builder_ = new MLGraphBuilder(this.context_);
    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});

    const bottleneck0 = await this.buildLinearBottleneck_(
        input, '0', false, 32, 'convRelu6');
    const bottleneck1 = await this.buildLinearBottleneck_(
        bottleneck0, '1', false, 96, 'dwiseRelu6');
    const bottleneck2 = await this.buildLinearBottleneck_(
        bottleneck1, '2', true, 144);
    const bottleneck3 = await this.buildLinearBottleneck_(
        bottleneck2, '3', false, 144, 'dwiseRelu6');
    const bottleneck4 = await this.buildLinearBottleneck_(
        bottleneck3, '4', true, 192);
    const bottleneck5 = await this.buildLinearBottleneck_(
        bottleneck4, '5', true, 192);
    const bottleneck6 = await this.buildLinearBottleneck_(
        bottleneck5, '6', false, 192, 'dwiseRelu6');
    const bottleneck7 = await this.buildLinearBottleneck_(
        bottleneck6, '7', true, 384);
    const bottleneck8 = await this.buildLinearBottleneck_(
        bottleneck7, '8', true, 384);
    const bottleneck9 = await this.buildLinearBottleneck_(
        bottleneck8, '9', true, 384);
    const bottleneck10 = await this.buildLinearBottleneck_(
        bottleneck9, '10', false, 384);
    const bottleneck11 = await this.buildLinearBottleneck_(
        bottleneck10, '11', true, 576);
    const bottleneck12 = await this.buildLinearBottleneck_(
        bottleneck11, '12', true, 576);
    const conv13Relu6 = await this.buildConv_(
        bottleneck12, ['expanded', '13']);
    const dwise13Relu6 = await this.buildConv_(
        conv13Relu6,
        ['expanded_depthwise', '13'],
        true,
        {groups: 576, strides: [2, 2]});
    const convLinear13 = await this.buildConv_(
        dwise13Relu6, ['expanded_project', '13'], false);

    const biasAdd0 = await this.buildConv_(
        conv13Relu6, ['BoxEncoding', '0'], false);
    const biasAdd3 = await this.buildConv_(
        conv13Relu6, ['Class', '0'], false);

    const bottleneck14 = await this.buildLinearBottleneck_(
        convLinear13, '14', true, 960);
    const bottleneck15 = await this.buildLinearBottleneck_(
        bottleneck14, '15', true, 960);
    const bottleneck16 = await this.buildLinearBottleneck_(
        bottleneck15, '16', false, 960);

    const conv17Relu6 = await this.buildConv_(
        bottleneck16, ['FeatureExtractor_MobilenetV2_Conv_1']);
    const biasAdd6 = await this.buildConv_(
        conv17Relu6, ['BoxEncoding', '1'], false);
    const biasAdd9 = await this.buildConv_(
        conv17Relu6, ['Class', '1'], false);

    const conv18Relu6 = await this.buildConv_(
        conv17Relu6, ['layer_19_1', '2_1x1_256']);
    const conv19Relu6 = await this.buildConv_(
        conv18Relu6, ['layer_19_2', '2_3x3_s2_512'], true, {strides: [2, 2]});
    const biasAdd12 = await this.buildConv_(
        conv19Relu6, ['BoxEncoding', '2'], false);
    const biasAdd15 = await this.buildConv_(
        conv19Relu6, ['Class', '2'], false);

    const conv20Relu6 = await this.buildConv_(
        conv19Relu6, ['layer_19_1', '3_1x1_128']);
    const conv21Relu6 = await this.buildConv_(
        conv20Relu6, ['layer_19_2', '3_3x3_s2_256'], true, {strides: [2, 2]});
    const biasAdd18 = await this.buildConv_(
        conv21Relu6, ['BoxEncoding', '3'], false);
    const biasAdd21 = await this.buildConv_(
        conv21Relu6, ['Class', '3'], false);

    const conv22Relu6 = await this.buildConv_(
        conv21Relu6, ['layer_19_1', '4_1x1_128']);
    const conv23Relu6 = await this.buildConv_(
        conv22Relu6, ['layer_19_2', '4_3x3_s2_256'], true, {strides: [2, 2]});
    const biasAdd24 = await this.buildConv_(
        conv23Relu6, ['BoxEncoding', '4'], false);
    const biasAdd27 = await this.buildConv_(
        conv23Relu6, ['Class', '4'], false);

    const conv24Relu6 = await this.buildConv_(
        conv23Relu6, ['layer_19_1', '5_1x1_64']);
    const conv25Relu6 = await this.buildConv_(
        conv24Relu6, ['layer_19_2', '5_3x3_s2_128'], true, {strides: [2, 2]});
    const biasAdd30 = await this.buildConv_(
        conv25Relu6, ['BoxEncoding', '5'], false);
    const biasAdd33 = await this.buildConv_(
        conv25Relu6, ['Class', '5'], false);

    return {biasAdd0, biasAdd3, biasAdd6, biasAdd9, biasAdd12, biasAdd15,
      biasAdd18, biasAdd21, biasAdd24, biasAdd27, biasAdd30, biasAdd33};
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
