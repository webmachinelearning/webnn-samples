
'use strict';

const nn = navigator.ml.getNeuralNetworkContext();

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

export class NSNet {
  constructor(url, batchSize, frames) {
    this.url_ = url;
    this.batchSize_ = batchSize;
    this.frames_ = frames;
    this.hiddenSize_ = 400;
    this.model_ = null;
    this.compilation_ = null;
  }

  async fetchData(fileName) {
    const response = await fetch(this.url_ + fileName);
    return new Float32Array(await response.arrayBuffer());
  }

  async load() {
    const weightData172 = await this.fetchData('172.bin');
    const biasDataFcIn0 = await this.fetchData('fc_in_0_bias.bin');
    const weightData192 = await this.fetchData('192.bin');
    const weightData193 = await this.fetchData('193.bin');
    const biasData194 = await this.fetchData('194.bin');
    const weightData212 = await this.fetchData('212.bin');
    const weightData213 = await this.fetchData('213.bin');
    const biasData214 = await this.fetchData('214.bin');
    const weightData215 = await this.fetchData('215.bin');
    const weightData216 = await this.fetchData('216.bin');
    const weightData217 = await this.fetchData('217.bin');
    const biasDataFcOut0 = await this.fetchData('fc_out_0_bias.bin');
    const biasDataFcOut2 = await this.fetchData('fc_out_2_bias.bin');
    const biasDataFcOut4 = await this.fetchData('fc_out_4_bias.bin');

    const builder = nn.createModelBuilder();
    
    const input = builder.input('input', {type: 'float32', dimensions: [this.batchSize_, this.frames_, 161]});
    const weight172 = builder.constant({type: 'float32', dimensions: [161, 400]}, weightData172);
    const matmul18 = builder.matmul(input, weight172);
    
    const biasFcIn0 = builder.constant({type: 'float32', dimensions: [400]}, biasDataFcIn0);
    const add19 = builder.add(matmul18, biasFcIn0);
    
    const relu20 = builder.relu(add19);
    const transpose31 = builder.transpose(relu20, {permutation: [1, 0, 2]});

    const weight192 = builder.constant({type: 'float32', dimensions: [1, 1200, 400]}, weightData192);
    const recurrentWeight193 = builder.constant({type: 'float32', dimensions: [1, 1200, 400]}, weightData193);
    const bias194 = builder.constant({type: 'float32', dimensions: [1, 1200]}, biasData194.subarray(0, 1200));
    const recurrentBias194 = builder.constant({type: 'float32', dimensions: [1, 1200]}, biasData194.subarray(1200));
    const [, gru93] = builder.gru(transpose31, weight192, recurrentWeight193, this.frames_, 400, {bias: bias194, recurrentBias: recurrentBias194, returnSequence: true});

    const squeeze95 = builder.squeeze(gru93, {axes: [1]});

    const weight212 = builder.constant({type: 'float32', dimensions: [1, 1200, 400]}, weightData212);
    const recurrentWeight213 = builder.constant({type: 'float32', dimensions: [1, 1200, 400]}, weightData213);
    const bias214 = builder.constant({type: 'float32', dimensions: [1, 1200]}, biasData214.subarray(0, 1200));
    const recurrentBias214 = builder.constant({type: 'float32', dimensions: [1, 1200]}, biasData214.subarray(1200));
    const [, gru156] = builder.gru(squeeze95, weight212, recurrentWeight213, this.frames_, 400, {bias: bias214, recurrentBias: recurrentBias214, returnSequence: true});

    const squeeze158 = builder.squeeze(gru156, {axes: [1]});
    const transpose159 = builder.transpose(squeeze158, {permutation: [1, 0, 2]});

    const weight215 = builder.constant({type: 'float32', dimensions: [400, 600]}, weightData215);
    const matmul161 = builder.matmul(transpose159, weight215);

    const biasFcOut0 = builder.constant({type: 'float32', dimensions: [600]}, biasDataFcOut0);
    const add162 = builder.add(matmul161, biasFcOut0);

    const relu163 = builder.relu(add162);

    const weight216 = builder.constant({type: 'float32', dimensions: [600, 600]}, weightData216);
    const matmul165 = builder.matmul(relu163, weight216);

    const biasFcOut2 = builder.constant({type: 'float32', dimensions: [600]}, biasDataFcOut2);
    const add166 = builder.add(matmul165, biasFcOut2);

    const relu167 = builder.relu(add166);

    const weight217 = builder.constant({type: 'float32', dimensions: [600, 161]}, weightData217);
    const matmul169 = builder.matmul(relu167, weight217);

    const biasFcOut4 = builder.constant({type: 'float32', dimensions: [161]}, biasDataFcOut4);
    const add170 = builder.add(matmul169, biasFcOut4);

    const output = builder.sigmoid(add170);

    this.model_ = builder.createModel({'output': output});
  }

  async compile(options) {
    this.compilation_ = await this.model_.compile(options);
  }

  async compute(inputBuffer) {
    const inputs = {input: {buffer: inputBuffer}};
    const outputs = await this.compilation_.compute(inputs);
    return outputs.output;
  }

  dispose() {
    this.compilation_.dispose();
  }
}
