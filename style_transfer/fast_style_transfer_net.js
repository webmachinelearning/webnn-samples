'use strict';

import {buildConstantByNpy} from '../common/utils.js';

/* eslint max-len: ["error", {"code": 130}] */

// Fast Style Transfer Baseline Model
export class FastStyleTransferNet {
  constructor() {
    this.builder_ = null;
    this.graph_ = null;
    this.constPow_ = null;
    this.constAdd_ = null;
    this.weightsUrl_ = '../test-data/models/fast_style_transfer_nchw/weights/';
    this.inputOptions = {
      inputDimensions: [1, 3, 540, 540],
      inputLayout: 'nchw',
    };
    this.outputDimensions = [1, 3, 540, 540];
  }

  buildInstanceNormalization_(conv2D, variableMul, variableAdd) {
    if ('instanceNormalization' in this.builder_) {
      return this.builder_.instanceNormalization(conv2D,
          {scale: this.builder_.squeeze(variableMul), bias: this.builder_.squeeze(variableAdd)});
    } else {
      const sub = this.builder_.sub(conv2D, this.builder_.reduceMean(conv2D, {axes: [2, 3], keepDimensions: true}));
      const reduceMean = this.builder_.reduceMean(this.builder_.mul(sub, sub), {axes: [2, 3], keepDimensions: true});
      const pow = this.builder_.pow(this.builder_.add(reduceMean, this.constAdd_), this.constPow_);
      const mul = this.builder_.mul(variableMul, this.builder_.div(sub, pow));
      return this.builder_.add(mul, variableAdd);
    }
  }

  async load(devicePreference, modelId) {
    const context = navigator.ml.createContext({devicePreference});
    this.builder_ = new MLGraphBuilder(context);
    const baseUrl = this.weightsUrl_ + modelId + '/';

    // Create constants by loading pre-trained data from .npy files.
    const weightConv0 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_read__0__cf__0_0.npy');
    const variableAdd0 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_1_read__1__cf__1_0.npy');
    const variableMul0 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_2_read__12__cf__12_0.npy');
    const weightConv1 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_3_read__23__cf__23_0.npy');
    const variableAdd1 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_4_read__34__cf__34_0.npy');
    const variableMul1 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_5_read__43__cf__43_0.npy');
    const weightConv2 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_6_read__44__cf__44_0.npy');
    const variableAdd2 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_7_read__45__cf__45_0.npy');
    const variableMul2 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_8_read__46__cf__46_0.npy');
    const weightConv3 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_9_read__47__cf__47_0.npy');
    const variableAdd3 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_10_read__2__cf__2_0.npy');
    const variableMul3 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_11_read__3__cf__3_0.npy');
    const weightConv4 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_12_read__4__cf__4_0.npy');
    const variableAdd4 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_13_read__5__cf__5_0.npy');
    const variableMul4 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_14_read__6__cf__6_0.npy');
    const weightConv5 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_15_read__7__cf__7_0.npy');
    const variableAdd5 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_16_read__8__cf__8_0.npy');
    const variableMul5 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_17_read__9__cf__9_0.npy');
    const weightConv6 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_18_read__10__cf__10_0.npy');
    const variableAdd6 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_19_read__11__cf__11_0.npy');
    const variableMul6 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_20_read__13__cf__13_0.npy');
    const weightConv7 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_21_read__14__cf__14_0.npy');
    const variableAdd7 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_22_read__15__cf__15_0.npy');
    const variableMul7 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_23_read__16__cf__16_0.npy');
    const weightConv8 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_24_read__17__cf__17_0.npy');
    const variableAdd8 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_25_read__18__cf__18_0.npy');
    const variableMul8 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_26_read__19__cf__19_0.npy');
    const weightConv9 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_27_read__20__cf__20_0.npy');
    const variableAdd9 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_28_read__21__cf__21_0.npy');
    const variableMul9 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_29_read__22__cf__22_0.npy');
    const weightConv10 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_30_read__24__cf__24_0.npy');
    const variableAdd10 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_31_read__25__cf__25_0.npy');
    const variableMul10 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_32_read__26__cf__26_0.npy');
    const weightConv11 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_33_read__27__cf__27_0.npy');
    const variableAdd11 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_34_read__28__cf__28_0.npy');
    const variableMul11 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_35_read__29__cf__29_0.npy');
    const weightConv12 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_36_read__30__cf__30_0.npy');
    const variableAdd12 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_37_read__31__cf__31_0.npy');
    const variableMul12 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_38_read__32__cf__32_0.npy');
    const weightConvTranspose0 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_39_read__33__cf__33_0.npy');
    const variableAdd13 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_40_read__35__cf__35_0.npy');
    const variableMul13 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_41_read__36__cf__36_0.npy');
    const weightConvTranspose1 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_42_read__37__cf__37_0.npy');
    const variableAdd14 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_43_read__38__cf__38_0.npy');
    const variableMul14 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_44_read__39__cf__39_0.npy');
    const weightConv13 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_45_read__40__cf__40_0.npy');
    const variableAdd15 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_46_read__41__cf__41_0.npy');
    const variableMul15 = await buildConstantByNpy(this.builder_, baseUrl + 'Variable_47_read__42__cf__42_0.npy');

    const padding1 = this.builder_.constant(
        {type: 'int32', dimensions: [4, 2]}, new Int32Array([0, 0, 0, 0, 1, 1, 1, 1]));
    const padding4 = this.builder_.constant(
        {type: 'int32', dimensions: [4, 2]}, new Int32Array([0, 0, 0, 0, 4, 4, 4, 4]));
    this.constAdd_ = this.builder_.constant(
        {type: 'float32', dimensions: [1]}, new Float32Array([9.999999717180685e-10]));
    this.constPow_ = this.builder_.constant(
        {type: 'float32', dimensions: [1]}, new Float32Array([0.5]));
    const constMul0 = this.builder_.constant(
        {type: 'float32', dimensions: [1]}, new Float32Array([150]));
    const constAdd0 = this.builder_.constant(
        {type: 'float32', dimensions: [1]}, new Float32Array([127.5]));
    // Build up the network.
    const input = this.builder_.input('input', {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const conv2D0 = this.builder_.conv2d(this.builder_.pad(input, padding4, {mode: 'reflection'}), weightConv0);

    const add0 = this.buildInstanceNormalization_(conv2D0, variableMul0, variableAdd0);
    const relu0 = this.builder_.relu(add0);
    const conv2D1 = this.builder_.conv2d(this.builder_.pad(relu0, padding1, {mode: 'reflection'}),
        weightConv1, {strides: [2, 2]});

    const add1 = this.buildInstanceNormalization_(conv2D1, variableMul1, variableAdd1);
    const relu1 = this.builder_.relu(add1);
    const conv2D2 = this.builder_.conv2d(this.builder_.pad(relu1, padding1, {mode: 'reflection'}),
        weightConv2, {strides: [2, 2]});

    const add2 = this.buildInstanceNormalization_(conv2D2, variableMul2, variableAdd2);
    const relu2 = this.builder_.relu(add2); // next input
    const conv2D3 = this.builder_.conv2d(this.builder_.pad(relu2, padding1, {mode: 'reflection'}), weightConv3);

    const add3 = this.buildInstanceNormalization_(conv2D3, variableMul3, variableAdd3);
    const relu3 = this.builder_.relu(add3);
    const conv2D4 = this.builder_.conv2d(this.builder_.pad(relu3, padding1, {mode: 'reflection'}), weightConv4);

    const add4 = this.buildInstanceNormalization_(conv2D4, variableMul4, variableAdd4);
    const add5 = this.builder_.add(relu2, add4); // next input
    const conv2D5 = this.builder_.conv2d(this.builder_.pad(add5, padding1, {mode: 'reflection'}), weightConv5);

    const add6 = this.buildInstanceNormalization_(conv2D5, variableMul5, variableAdd5);
    const relu4 = this.builder_.relu(add6);
    const conv2D6 = this.builder_.conv2d(this.builder_.pad(relu4, padding1, {mode: 'reflection'}), weightConv6);

    const add7 = this.buildInstanceNormalization_(conv2D6, variableMul6, variableAdd6);
    const add8 = this.builder_.add(add5, add7); // next input
    const conv2D7 = this.builder_.conv2d(this.builder_.pad(add8, padding1, {mode: 'reflection'}), weightConv7);

    const add9 = this.buildInstanceNormalization_(conv2D7, variableMul7, variableAdd7);
    const relu5 = this.builder_.relu(add9);
    const conv2D8 = this.builder_.conv2d(this.builder_.pad(relu5, padding1, {mode: 'reflection'}), weightConv8);

    const add10 = this.buildInstanceNormalization_(conv2D8, variableMul8, variableAdd8);
    const add11 = this.builder_.add(add8, add10); // next input
    const conv2D9 = this.builder_.conv2d(this.builder_.pad(add11, padding1, {mode: 'reflection'}), weightConv9);

    const add12 = this.buildInstanceNormalization_(conv2D9, variableMul9, variableAdd9);
    const relu6 = this.builder_.relu(add12);
    const conv2D10 = this.builder_.conv2d(this.builder_.pad(relu6, padding1, {mode: 'reflection'}), weightConv10);

    const add13 = this.buildInstanceNormalization_(conv2D10, variableMul10, variableAdd10);
    const add14 = this.builder_.add(add11, add13); // next input
    const conv2D11 = this.builder_.conv2d(this.builder_.pad(add14, padding1, {mode: 'reflection'}), weightConv11);

    const add15 = this.buildInstanceNormalization_(conv2D11, variableMul11, variableAdd11);
    const relu7 = this.builder_.relu(add15);
    const conv2D12 = this.builder_.conv2d(this.builder_.pad(relu7, padding1, {mode: 'reflection'}), weightConv12);

    const add16 = this.buildInstanceNormalization_(conv2D12, variableMul12, variableAdd12);
    const add17 = this.builder_.add(add14, add16);
    const convTranspose0 = this.builder_.convTranspose2d(add17, weightConvTranspose0,
        {strides: [2, 2], outputSizes: [270, 270]});

    const add18 = this.buildInstanceNormalization_(convTranspose0, variableMul13, variableAdd13);
    const relu8 = this.builder_.relu(add18);
    const convTranspose1 = this.builder_.convTranspose2d(relu8, weightConvTranspose1,
        {strides: [2, 2], outputSizes: [540, 540]});

    const add19 = this.buildInstanceNormalization_(convTranspose1, variableMul14, variableAdd14);
    const relu9 = this.builder_.relu(add19);
    const conv2D13 = this.builder_.conv2d(this.builder_.pad(relu9, padding4, {mode: 'reflection'}), weightConv13);

    const add20 = this.buildInstanceNormalization_(conv2D13, variableMul15, variableAdd15);
    return this.builder_.add(this.builder_.mul(this.builder_.tanh(add20), constMul0), constAdd0);
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
