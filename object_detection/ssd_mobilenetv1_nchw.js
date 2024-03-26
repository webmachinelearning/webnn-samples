'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

// SSD MobileNet V1 model with 'nchw' layout, trained on the COCO dataset.
export class SsdMobilenetV1Nchw {
  constructor() {
    this.context_ = null;
    this.deviceType_ = null;
    this.model_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/ssd_mobilenetv1_nchw/weights';
    // Shares the same bias files with 'nhwc' layout
    this.biasUrl_ = weightsOrigin() +
      '/test-data/models/ssd_mobilenetv1_nhwc/weights';
    this.inputOptions = {
      inputLayout: 'nchw',
      labelUrl: './labels/coco_classes.txt',
      margin: [1, 1, 1, 1],
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputDimensions: [1, 3, 300, 300],
    };
  }

  async buildConv_(input, nameArray, relu6 = true, options = {}) {
    // nameArray: 0: keyword, 1: indice, 2: weightSuffix, 3: biasSuffix
    let prefix = '';
    let weightSuffix = '_mul_1.npy';
    let biasSuffix = `_sub__${nameArray[3]}.npy`;

    if (nameArray[0].includes('depthwise')) {
      prefix += `/FeatureExtractor_MobilenetV1_MobilenetV1_Conv2d_\
${nameArray[1]}_depthwise_BatchNorm_batchnorm`;
      weightSuffix = `_mul__${nameArray[2]}.npy`;
    } else if (nameArray[0].includes('pointwise')) {
      if (nameArray[0].includes('_')) {
        prefix += `/FeatureExtractor_MobilenetV1_Conv2d_13_\
${nameArray[0]}_Conv2d_${nameArray[1]}_BatchNorm_batchnorm`;
      } else {
        prefix += `/FeatureExtractor_MobilenetV1_MobilenetV1_Conv2d_\
${nameArray[1]}_pointwise_BatchNorm_batchnorm`;
      }
    } else if (nameArray[0].includes('Class')) {
      prefix += `/BoxPredictor_${nameArray[1]}_ClassPredictor`;
      weightSuffix = '_Conv2D.npy';
      biasSuffix = `_biases_read__${nameArray[3]}.npy`;
    } else if (nameArray[0].includes('BoxEncoding')) {
      prefix += `/BoxPredictor_${nameArray[1]}_BoxEncodingPredictor`;
      weightSuffix = '_Conv2D.npy';
      biasSuffix = `_biases_read__${nameArray[3]}.npy`;
    } else {
      prefix += `/FeatureExtractor_MobilenetV1_MobilenetV1_Conv2d_\
${nameArray[1]}_BatchNorm_batchnorm`;
    }

    const weightsName = this.weightsUrl_ + prefix + weightSuffix;
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = this.biasUrl_ + prefix + biasSuffix;
    const bias = await buildConstantByNpy(this.builder_, biasName);
    options.padding = computePadding2DForAutoPad(
        /* nchw */[input.shape()[2], input.shape()[3]],
        /* oihw */[weights.shape()[2], weights.shape()[3]],
        options.strides, options.dilations, 'same-upper');
    options.bias = bias;
    if (relu6) {
      // TODO: Set clamp activation to options once it's supported in
      // WebNN DML backend.
      // Implement `clip` by `clamp` of  WebNN API
      if (this.deviceType_ == 'gpu') {
        return this.builder_.clamp(
            this.builder_.conv2d(input, weights, options),
            {minValue: 0, maxValue: 6});
      } else {
        options.activation = this.builder_.clamp({minValue: 0, maxValue: 6});
      }
    }
    return this.builder_.conv2d(input, weights, options);
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
    const strides = [2, 2];
    const conv0 = await this.buildConv_(
        input, ['', '0', '', '165__cf__168'],
        true, {strides});
    const dwise0 = await this.buildConv_(
        conv0, ['depthwise', '1', '161__cf__164', '162__cf__165'],
        true, {groups: 32});
    const conv1 = await this.buildConv_(
        dwise0, ['pointwise', '1', '', '159__cf__162']);
    const dwise1 = await this.buildConv_(
        conv1, ['depthwise', '2', '155__cf__158', '156__cf__159'],
        true, {strides, groups: 64});
    const conv2 = await this.buildConv_(
        dwise1, ['pointwise', '2', '', '153__cf__156']);
    const dwise2 = await this.buildConv_(
        conv2, ['depthwise', '3', '149__cf__152', '150__cf__153'],
        true, {groups: 128});
    const conv3 = await this.buildConv_(
        dwise2, ['pointwise', '3', '', '147__cf__150']);
    const dwise3 = await this.buildConv_(
        conv3, ['depthwise', '4', '143__cf__146', '144__cf__147'],
        true, {strides, groups: 128});
    const conv4 = await this.buildConv_(
        dwise3, ['pointwise', '4', '', '141__cf__144']);
    const dwise4 = await this.buildConv_(
        conv4, ['depthwise', '5', '137__cf__140', '138__cf__141'],
        true, {groups: 256});
    const conv5 = await this.buildConv_(
        dwise4, ['pointwise', '5', '', '135__cf__138']);
    const dwise5 = await this.buildConv_(
        conv5, ['depthwise', '6', '131__cf__134', '132__cf__135'],
        true, {strides, groups: 256});
    const conv6 = await this.buildConv_(
        dwise5, ['pointwise', '6', '', '129__cf__132']);
    const dwise6 = await this.buildConv_(
        conv6, ['depthwise', '7', '125__cf__128', '126__cf__129'],
        true, {groups: 512});
    const conv7 = await this.buildConv_(
        dwise6, ['pointwise', '7', '', '123__cf__126']);
    const dwise7 = await this.buildConv_(
        conv7, ['depthwise', '8', '119__cf__122', '120__cf__123'],
        true, {groups: 512});
    const conv8 = await this.buildConv_(
        dwise7, ['pointwise', '8', '', '117__cf__120']);
    const dwise8 = await this.buildConv_(
        conv8, ['depthwise', '9', '113__cf__116', '114__cf__117'],
        true, {groups: 512});
    const conv9 = await this.buildConv_(
        dwise8, ['pointwise', '9', '', '111__cf__114']);
    const dwise9 = await this.buildConv_(
        conv9, ['depthwise', '10', '107__cf__110', '108__cf__111'],
        true, {groups: 512});
    const conv10 = await this.buildConv_(
        dwise9, ['pointwise', '10', '', '105__cf__108']);
    const dwise10 = await this.buildConv_(
        conv10, ['depthwise', '11', '101__cf__104', '102__cf__105'],
        true, {groups: 512});
    const conv11 = await this.buildConv_(
        dwise10, ['pointwise', '11', '', '99__cf__102']);

    const dwise11 = await this.buildConv_(
        conv11, ['depthwise', '12', '95__cf__98', '96__cf__99'],
        true, {strides, groups: 512});
    const conv12 = await this.buildConv_(
        dwise11, ['pointwise', '12', '', '93__cf__96']);
    const dwise12 = await this.buildConv_(
        conv12, ['depthwise', '13', '89__cf__92', '90__cf__93'],
        true, {groups: 1024});
    const conv13 = await this.buildConv_(
        dwise12, ['pointwise', '13', '', '87__cf__90']);

    const conv14 = await this.buildConv_(
        conv13, ['pointwise_1', '2_1x1_256', '', '84__cf__87']);
    const conv15 = await this.buildConv_(
        conv14, ['pointwise_2', '2_3x3_s2_512', '', '81__cf__84'],
        true, {strides});
    const conv16 = await this.buildConv_(
        conv15, ['pointwise_1', '3_1x1_128', '', '78__cf__81']);
    const conv17 = await this.buildConv_(
        conv16, ['pointwise_2', '3_3x3_s2_256', '', '75__cf__78'],
        true, {strides});
    const conv18 = await this.buildConv_(
        conv17, ['pointwise_1', '4_1x1_128', '', '72__cf__75']);
    const conv19 = await this.buildConv_(
        conv18, ['pointwise_2', '4_3x3_s2_256', '', '69__cf__72'],
        true, {strides});
    const conv20 = await this.buildConv_(
        conv19, ['pointwise_1', '5_1x1_64', '', '66__cf__69']);
    const conv21 = await this.buildConv_(
        conv20, ['pointwise_2', '5_3x3_s2_128', '', '63__cf__66'],
        true, {strides});

    // First concatenation
    const conv22 = await this.buildConv_(
        conv11, ['BoxEncoding', '0', '', '177__cf__180'], false);
    const reshape0 = this.builder_.reshape(
        this.builder_.transpose(conv22, {permutation: [0, 2, 3, 1]}),
        [1, 1083, 1, 4]);
    const conv23 = await this.buildConv_(
        conv13, ['BoxEncoding', '1', '', '175__cf__178'], false);
    const reshape1 = this.builder_.reshape(
        this.builder_.transpose(conv23, {permutation: [0, 2, 3, 1]}),
        [1, 600, 1, 4]);
    const conv24 = await this.buildConv_(
        conv15, ['BoxEncoding', '2', '', '173__cf__176'], false);
    const reshape2 = this.builder_.reshape(
        this.builder_.transpose(conv24, {permutation: [0, 2, 3, 1]}),
        [1, 150, 1, 4]);
    const conv25 = await this.buildConv_(
        conv17, ['BoxEncoding', '3', '', '171__cf__174'], false);
    const reshape3 = this.builder_.reshape(
        this.builder_.transpose(conv25, {permutation: [0, 2, 3, 1]}),
        [1, 54, 1, 4]);
    const conv26 = await this.buildConv_(
        conv19, ['BoxEncoding', '4', '', '169__cf__172'], false);
    const reshape4 = this.builder_.reshape(
        this.builder_.transpose(conv26, {permutation: [0, 2, 3, 1]}),
        [1, 24, 1, 4]);
    const conv27 = await this.buildConv_(
        conv21, ['BoxEncoding', '5', '', '167__cf__170'], false);
    const reshape5 = this.builder_.reshape(
        this.builder_.transpose(conv27, {permutation: [0, 2, 3, 1]}),
        [1, 6, 1, 4]);
    const concat0 = this.builder_.concat(
        [reshape0, reshape1, reshape2, reshape3, reshape4, reshape5], 1);

    // Second concatenation
    const conv28 = await this.buildConv_(
        conv11, ['Class', '0', '', '51__cf__54'], false);
    const reshape6 = this.builder_.reshape(
        this.builder_.transpose(conv28, {permutation: [0, 2, 3, 1]}),
        [1, 1083, 91]);
    const conv29 = await this.buildConv_(
        conv13, ['Class', '1', '', '49__cf__52'], false);
    const reshape7 = this.builder_.reshape(
        this.builder_.transpose(conv29, {permutation: [0, 2, 3, 1]}),
        [1, 600, 91]);
    const conv30 = await this.buildConv_(
        conv15, ['Class', '2', '', '47__cf__50'], false);
    const reshape8 = this.builder_.reshape(
        this.builder_.transpose(conv30, {permutation: [0, 2, 3, 1]}),
        [1, 150, 91]);
    const conv31 = await this.buildConv_(
        conv17, ['Class', '3', '', '45__cf__48'], false);
    const reshape9 = this.builder_.reshape(
        this.builder_.transpose(conv31, {permutation: [0, 2, 3, 1]}),
        [1, 54, 91]);
    const conv32 = await this.buildConv_(
        conv19, ['Class', '4', '', '43__cf__46'], false);
    const reshape10 = this.builder_.reshape(
        this.builder_.transpose(conv32, {permutation: [0, 2, 3, 1]}),
        [1, 24, 91]);
    const conv33 = await this.buildConv_(
        conv21, ['Class', '5', '', '41__cf__44'], false);
    const reshape11 = this.builder_.reshape(
        this.builder_.transpose(conv33, {permutation: [0, 2, 3, 1]}),
        [1, 6, 91]);
    const concat1 = this.builder_.concat(
        [reshape6, reshape7, reshape8, reshape9, reshape10, reshape11], 1);

    return {'boxes': concat0, 'scores': concat1};
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
