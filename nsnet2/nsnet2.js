
'use strict';

import {buildConstantByNpy} from '../common/utils.js';

/* eslint max-len: ["error", { "code": 130 }] */

// Noise Suppression Net 2 (NSNet2) Baseline Model for Deep Noise Suppression Challenge (DNS) 2021.
export class NSNet2 {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.frameSize = 161;
    this.hiddenSize = 400;
  }

  async load(contextOptions, baseUrl, batchSize, frames) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    // Create constants by loading pre-trained data from .npy files.
    const weight172 = await buildConstantByNpy(this.builder_, baseUrl + '172.npy');
    const biasFcIn0 = await buildConstantByNpy(this.builder_, baseUrl + 'fc_in_0_bias.npy');
    const weight192 = await buildConstantByNpy(this.builder_, baseUrl + '192.npy');
    const recurrentWeight193 = await buildConstantByNpy(this.builder_, baseUrl + '193.npy');
    const bias194 = await buildConstantByNpy(this.builder_, baseUrl + '194_0.npy');
    const recurrentBias194 = await buildConstantByNpy(this.builder_, baseUrl + '194_1.npy');
    const weight212 = await buildConstantByNpy(this.builder_, baseUrl + '212.npy');
    const recurrentWeight213 = await buildConstantByNpy(this.builder_, baseUrl + '213.npy');
    const bias214 = await buildConstantByNpy(this.builder_, baseUrl + '214_0.npy');
    const recurrentBias214 = await buildConstantByNpy(this.builder_, baseUrl + '214_1.npy');
    const weight215 = await buildConstantByNpy(this.builder_, baseUrl + '215.npy');
    const biasFcOut0 = await buildConstantByNpy(this.builder_, baseUrl + 'fc_out_0_bias.npy');
    const weight216 = await buildConstantByNpy(this.builder_, baseUrl + '216.npy');
    const biasFcOut2 = await buildConstantByNpy(this.builder_, baseUrl + 'fc_out_2_bias.npy');
    const weight217 = await buildConstantByNpy(this.builder_, baseUrl + '217.npy');
    const biasFcOut4 = await buildConstantByNpy(this.builder_, baseUrl + 'fc_out_4_bias.npy');
    // Build up the network.
    const input = this.builder_.input('input', {
      type: 'float32',
      dataType: 'float32',
      dimensions: [batchSize, frames, this.frameSize],
    });
    const relu20 = this.builder_.relu(this.builder_.add(this.builder_.matmul(input, weight172), biasFcIn0));
    const transpose31 = this.builder_.transpose(relu20, {permutation: [1, 0, 2]});
    const initialState92 = this.builder_.input('initialState92', {
      type: 'float32',
      dataType: 'float32',
      dimensions: [1, batchSize, this.hiddenSize],
    });
    const [gru94, gru93] = this.builder_.gru(transpose31, weight192, recurrentWeight193, frames, this.hiddenSize,
        {bias: bias194, recurrentBias: recurrentBias194, initialHiddenState: initialState92, returnSequence: true});
    // Use reshape to implement squeeze(gru93, {axes: [1]});
    const squeeze95Shape = gru93.shape();
    squeeze95Shape.splice(1, 1);
    const squeeze95 = this.builder_.reshape(gru93, squeeze95Shape);
    const initialState155 = this.builder_.input('initialState155', {
      type: 'float32',
      dataType: 'float32',
      dimensions: [1, batchSize, this.hiddenSize],
    });
    const [gru157, gru156] = this.builder_.gru(squeeze95, weight212, recurrentWeight213, frames, this.hiddenSize,
        {bias: bias214, recurrentBias: recurrentBias214, initialHiddenState: initialState155, returnSequence: true});
    // Use reshape to implement squeeze(gru156, {axes: [1]});
    const squeeze158Shape = gru156.shape();
    squeeze158Shape.splice(1, 1);
    const squeeze158 = this.builder_.reshape(gru156, squeeze158Shape);
    const transpose159 = this.builder_.transpose(squeeze158, {permutation: [1, 0, 2]});
    const relu163 = this.builder_.relu(this.builder_.add(this.builder_.matmul(transpose159, weight215), biasFcOut0));
    const relu167 = this.builder_.relu(this.builder_.add(this.builder_.matmul(relu163, weight216), biasFcOut2));
    const output = this.builder_.sigmoid(this.builder_.add(this.builder_.matmul(relu167, weight217), biasFcOut4));
    return {output, gru94, gru157};
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build(outputOperand);
  }

  async compute(inputBuffer, initialState92Buffer, initialState155Buffer, outputBuffer, gru94Buffer, gru157Buffer) {
    const inputs = {
      'input': inputBuffer,
      'initialState92': initialState92Buffer,
      'initialState155': initialState155Buffer,
    };
    const outputs = {
      'output': outputBuffer.slice(),
      'gru94': gru94Buffer.slice(),
      'gru157': gru157Buffer.slice(),
    };
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results.outputs;
  }
}
