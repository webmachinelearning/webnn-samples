'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';
const strides = [2, 2];
const autoPad = 'same-upper';

/* eslint-disable camelcase */

// FaceNet model with 'nchw' layout.
export class FaceNetNchw {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/facenet_nchw/weights';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5, 127.5],
      channelScheme: 'BGR',
      inputLayout: 'nchw',
      inputDimensions: [1, 3, 160, 160],
    };
    this.postOptions = {
      distanceMetric: 'euclidean',
      threshold: 1.26,
    };
  }

  async buildConv_(
      input, weightsSuffix, biasPrefix, options = undefined, relu = true) {
    const weightsName =
        `${this.weightsUrl_}/const_fold_opt__${weightsSuffix}.npy`;
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = `${this.weightsUrl_}/${biasPrefix}_Conv2D_bias.npy`;
    const bias = await buildConstantByNpy(this.builder_, biasName);
    if (options !== undefined) {
      options.bias = bias;
    } else {
      options = {
        bias: bias,
      };
    }
    if (relu) {
      options.activation = this.builder_.relu();
    }
    // WebNN spec drops autoPad support, compute the explicit padding instead.
    if (options.autoPad == 'same-upper') {
      options.padding =
        computePadding2DForAutoPad(
            /* nchw */[input.shape()[2], input.shape()[3]],
            /* oihw */[weights.shape()[2], weights.shape()[3]],
            options.strides, options.dilations, options.autoPad);
    }
    return this.builder_.conv2d(input, weights, options);
  }

  async buildBlock35_(input, indice, weightsSuffixes) {
    const branch0 = await this.buildConv_(
        input, weightsSuffixes[0], `Block35_${indice}_Branch_0_Conv2d_1x1`);
    const branch1_0 = await this.buildConv_(
        input, weightsSuffixes[1], `Block35_${indice}_Branch_1_Conv2d_0a_1x1`);
    const branch1_1 = await this.buildConv_(
        branch1_0, weightsSuffixes[2],
        `Block35_${indice}_Branch_1_Conv2d_0b_3x3`, {autoPad});
    const branch2_0 = await this.buildConv_(
        input, weightsSuffixes[3], `Block35_${indice}_Branch_2_Conv2d_0a_1x1`);
    const branch2_1 = await this.buildConv_(
        branch2_0, weightsSuffixes[4],
        `Block35_${indice}_Branch_2_Conv2d_0b_3x3`, {autoPad});
    const branch2_2 = await this.buildConv_(
        branch2_1, weightsSuffixes[5],
        `Block35_${indice}_Branch_2_Conv2d_0c_3x3`, {autoPad});

    const concat = this.builder_.concat([branch0, branch1_1, branch2_2], 1);
    const conv = await this.buildConv_(
        concat, weightsSuffixes[6], `Block35_${indice}_Conv2d_1x1`, {}, false);

    return this.builder_.relu(this.builder_.add(input, conv));
  }

  async buildBlock17_(input, indice, weightsSuffixes) {
    const branch0 = await this.buildConv_(
        input, weightsSuffixes[0], `Block17_${indice}_Branch_0_Conv2d_1x1`);
    const branch1_0 = await this.buildConv_(
        input, weightsSuffixes[1], `Block17_${indice}_Branch_1_Conv2d_0a_1x1`);
    const branch1_1 = await this.buildConv_(
        branch1_0, weightsSuffixes[2],
        `Block17_${indice}_Branch_1_Conv2d_0b_1x7`, {autoPad});
    const branch1_2 = await this.buildConv_(
        branch1_1, weightsSuffixes[3],
        `Block17_${indice}_Branch_1_Conv2d_0c_7x1`, {autoPad});

    const concat = this.builder_.concat([branch0, branch1_2], 1);
    const conv = await this.buildConv_(
        concat, weightsSuffixes[4], `Block17_${indice}_Conv2d_1x1`, {}, false);

    return this.builder_.relu(this.builder_.add(input, conv));
  }

  async buildBlock8_(input, indice, weightsSuffixes, relu = true) {
    const branch0 = await this.buildConv_(
        input, weightsSuffixes[0], `Block8_${indice}_Branch_0_Conv2d_1x1`);
    const branch1_0 = await this.buildConv_(
        input, weightsSuffixes[1], `Block8_${indice}_Branch_1_Conv2d_0a_1x1`);
    const branch1_1 = await this.buildConv_(
        branch1_0, weightsSuffixes[2],
        `Block8_${indice}_Branch_1_Conv2d_0b_1x3`, {autoPad});
    const branch1_2 = await this.buildConv_(
        branch1_1, weightsSuffixes[3],
        `Block8_${indice}_Branch_1_Conv2d_0c_3x1`, {autoPad});

    const concat = this.builder_.concat([branch0, branch1_2], 1);
    const conv = await this.buildConv_(
        concat, weightsSuffixes[4], `Block8_${indice}_Conv2d_1x1`, {}, false);

    let result = this.builder_.add(input, conv);

    if (relu) {
      result = this.builder_.relu(result);
    }
    return result;
  }

  async buildGemm_(input) {
    const weights = await buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/const_fold_opt__888.npy`);
    const bias = await buildConstantByNpy(this.builder_,
        `${this.weightsUrl_}/Bottleneck_MatMul_bias.npy`);
    return this.builder_.gemm(input, weights, {c: bias});
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const input = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: this.inputOptions.inputDimensions,
    });

    const poolOptions = {windowDimensions: [3, 3], strides};

    const conv0 = await this.buildConv_(
        input, '1070', 'Conv2d_1a_3x3', {strides});
    const conv1 = await this.buildConv_(conv0, '1146', 'Conv2d_2a_3x3');
    const conv2 = await this.buildConv_(
        conv1, '1068', 'Conv2d_2b_3x3', {autoPad});

    const pool0 = this.builder_.maxPool2d(conv2, poolOptions);

    const conv3 = await this.buildConv_(pool0, '1058', 'Conv2d_3b_1x1');
    const conv4 = await this.buildConv_(conv3, '1045', 'Conv2d_4a_3x3');
    const conv5 = await this.buildConv_(
        conv4, '976', 'Conv2d_4b_3x3', {strides});

    // Block 35
    const block35_1 = await this.buildBlock35_(
        conv5, 1, ['930', '1073', '912', '1134', '1127', '921', '959']);
    const block35_2 = await this.buildBlock35_(
        block35_1, 2, ['1005', '1036', '902', '925', '1016', '1133', '1059']);
    const block35_3 = await this.buildBlock35_(
        block35_2, 3, ['1046', '1115', '1020', '1091', '958', '948', '891']);
    const block35_4 = await this.buildBlock35_(
        block35_3, 4, ['1053', '1151', '919', '961', '1111', '1124', '900']);
    const block35_5 = await this.buildBlock35_(
        block35_4, 5, ['1083', '1100', '1015', '1082', '1099', '1125', '1081']);

    // Mixed 6a branches
    const mixed6a_branch0 = await this.buildConv_(
        block35_5, '1079', 'Mixed_6a_Branch_0_Conv2d_1a_3x3', {strides});
    const mixed6a_pool = this.builder_.maxPool2d(block35_5, poolOptions);
    const mixed6a_branch1_0 = await this.buildConv_(
        block35_5, '1072', 'Mixed_6a_Branch_1_Conv2d_0a_1x1');
    const mixed6a_branch1_1 = await this.buildConv_(
        mixed6a_branch1_0, '1022',
        'Mixed_6a_Branch_1_Conv2d_0b_3x3', {autoPad});
    const mixed6a_branch1_2 = await this.buildConv_(
        mixed6a_branch1_1, '1012',
        'Mixed_6a_Branch_1_Conv2d_1a_3x3', {strides});
    const mixed6a = this.builder_.concat(
        [mixed6a_branch0, mixed6a_branch1_2, mixed6a_pool], 1);

    // Block 17
    const block17_1 = await this.buildBlock17_(
        mixed6a, 1, ['1119', '1112', '989', '1095', '952']);
    const block17_2 = await this.buildBlock17_(
        block17_1, 2, ['947', '918', '906', '1069', '1148']);
    const block17_3 = await this.buildBlock17_(
        block17_2, 3, ['1087', '932', '1028', '1150', '1137']);
    const block17_4 = await this.buildBlock17_(
        block17_3, 4, ['1089', '1039', '1001', '1135', '1129']);
    const block17_5 = await this.buildBlock17_(
        block17_4, 5, ['1123', '898', '1090', '1105', '920']);
    const block17_6 = await this.buildBlock17_(
        block17_5, 6, ['935', '998', '1143', '1061', '1049']);
    const block17_7 = await this.buildBlock17_(
        block17_6, 7, ['1118', '1023', '1013', '1092', '1093']);
    const block17_8 = await this.buildBlock17_(
        block17_7, 8, ['1141', '1024', '1140', '929', '1120']);
    const block17_9 = await this.buildBlock17_(
        block17_8, 9, ['1067', '908', '1142', '1003', '1074']);
    const block17_10 = await this.buildBlock17_(
        block17_9, 10, ['1078', '913', '1138', '982', '967']);

    // Mixed 7a branches
    const mixed7a_pool = this.builder_.maxPool2d(block17_10, poolOptions);
    const mixed7a_branch0_0 = await this.buildConv_(
        block17_10, '1103', 'Mixed_7a_Branch_0_Conv2d_0a_1x1');
    const mixed7a_branch0_1 = await this.buildConv_(
        mixed7a_branch0_0, '1066',
        'Mixed_7a_Branch_0_Conv2d_1a_3x3', {strides});
    const mixed7a_branch1_0 = await this.buildConv_(
        block17_10, '1056', 'Mixed_7a_Branch_1_Conv2d_0a_1x1');
    const mixed7a_branch1_1 = await this.buildConv_(
        mixed7a_branch1_0, '1043',
        'Mixed_7a_Branch_1_Conv2d_1a_3x3', {strides});
    const mixed7a_branch2_0 = await this.buildConv_(
        block17_10, '1145', 'Mixed_7a_Branch_2_Conv2d_0a_1x1');
    const mixed7a_branch2_1 = await this.buildConv_(
        mixed7a_branch2_0, '1077',
        'Mixed_7a_Branch_2_Conv2d_0b_3x3', {autoPad});
    const mixed7a_branch2_2 = await this.buildConv_(
        mixed7a_branch2_1, '897',
        'Mixed_7a_Branch_2_Conv2d_1a_3x3', {strides});
    const mixed7a = this.builder_.concat(
        [mixed7a_branch0_1, mixed7a_branch1_1,
          mixed7a_branch2_2, mixed7a_pool], 1);

    // Block 8
    const block8_1 = await this.buildBlock8_(
        mixed7a, 1, ['1065', '1126', '1110', '1116', '1107']);
    const block8_2 = await this.buildBlock8_(
        block8_1, 2, ['999', '1144', '1102', '1097', '981']);
    const block8_3 = await this.buildBlock8_(
        block8_2, 3, ['962', '934', '1064', '1052', '1042']);
    const block8_4 = await this.buildBlock8_(
        block8_3, 4, ['915', '957', '1000', '1034', '1071']);
    const block8_5 = await this.buildBlock8_(
        block8_4, 5, ['1139', '1114', '1106', '997', '970']);
    const block8_6 = await this.buildBlock8_(
        block8_5, 6, ['977', '1104', '978', '1080', '1086'], false);

    const averagePool = this.builder_.averagePool2d(block8_6);
    // Use reshape to implement squeeze(averagePool, {axes: [2, 3]});
    const squeezed_shape = averagePool.shape();
    squeezed_shape.splice(2, 2);
    const squeeze = this.builder_.reshape(averagePool, squeezed_shape);
    const gemm = await this.buildGemm_(squeeze);
    // L2Normalization will be handled in post-processing
    return gemm;
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

  async compute(inputBuffer, outputs) {
    const inputs = {'input': inputBuffer};
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results;
  }
}
