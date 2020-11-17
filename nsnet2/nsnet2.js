
'use strict';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

export class NSNet2 {
  constructor() {
    this.model_ = null;
    this.compilation_ = null;
  }

  async load(url, batchSize, frames) {
    // Fetch and verify buffer
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const WEIGHTS_FILE_SIZE = 10750244;
    if (buffer.byteLength !== WEIGHTS_FILE_SIZE) {
      throw new Error('Incorrect weights file');
    }

    // Constant shapes and sizes
    const hiddenSize = 400;
    const weight172Shape = [161, hiddenSize];
    const biasFcIn0Shape = [hiddenSize];
    const weight192Shape = [1, 3 * hiddenSize, hiddenSize];
    const recurrentWeight193Shape = [1, 3 * hiddenSize, hiddenSize];
    const bias194Shape = [1, 3 * hiddenSize];
    const recurrentBias194Shape = [1, 3 * hiddenSize];
    const weight212Shape = [1, 3 * hiddenSize, hiddenSize];
    const recurrentWeight213Shape = [1, 3 * hiddenSize, hiddenSize];
    const bias214Shape = [1, 3 * hiddenSize];
    const recurrentBias214Shape = [1, 3 * hiddenSize];
    const weight215Shape = [hiddenSize, 600];
    const biasFcOut0Shape = [600];
    const weight216Shape = [600, 600];
    const biasFcOut2Shape = [600];
    const weight217Shape = [600, 161];
    const biasFcOut4Shape = [161];

    // Load pre-trained constant data and initializers
    let offset = 0;
    const weight172Data = new Float32Array(
        buffer, offset, sizeOfShape(weight172Shape));
    offset += sizeOfShape(weight172Shape) * Float32Array.BYTES_PER_ELEMENT;
    const biasFcIn0Data = new Float32Array(
        buffer, offset, sizeOfShape(biasFcIn0Shape));
    offset += sizeOfShape(biasFcIn0Shape) * Float32Array.BYTES_PER_ELEMENT;
    const weight192Data = new Float32Array(
        buffer, offset, sizeOfShape(weight192Shape));
    offset += sizeOfShape(weight192Shape) * Float32Array.BYTES_PER_ELEMENT;
    const recurrentWeight193Data = new Float32Array(
        buffer, offset, sizeOfShape(recurrentWeight193Shape));
    offset += sizeOfShape(recurrentWeight193Shape) *
        Float32Array.BYTES_PER_ELEMENT;
    const bias194Data = new Float32Array(
        buffer, offset, sizeOfShape(bias194Shape));
    offset += sizeOfShape(bias194Shape) * Float32Array.BYTES_PER_ELEMENT;
    const recurrentBias194Data = new Float32Array(
        buffer, offset, sizeOfShape(recurrentBias194Shape));
    offset += sizeOfShape(recurrentBias194Shape) *
        Float32Array.BYTES_PER_ELEMENT;
    const weight212Data = new Float32Array(
        buffer, offset, sizeOfShape(weight212Shape));
    offset += sizeOfShape(weight212Shape) * Float32Array.BYTES_PER_ELEMENT;
    const recurrentWeight213Data = new Float32Array(
        buffer, offset, sizeOfShape(recurrentWeight213Shape));
    offset += sizeOfShape(recurrentWeight213Shape) *
        Float32Array.BYTES_PER_ELEMENT;
    const bias214Data = new Float32Array(
        buffer, offset, sizeOfShape(bias214Shape));
    offset += sizeOfShape(bias214Shape) * Float32Array.BYTES_PER_ELEMENT;
    const recurrentBias214Data = new Float32Array(
        buffer, offset, sizeOfShape(recurrentBias214Shape));
    offset += sizeOfShape(recurrentBias214Shape) *
        Float32Array.BYTES_PER_ELEMENT;
    const weight215Data = new Float32Array(
        buffer, offset, sizeOfShape(weight215Shape));
    offset += sizeOfShape(weight215Shape) * Float32Array.BYTES_PER_ELEMENT;
    const biasFcOut0Data = new Float32Array(
        buffer, offset, sizeOfShape(biasFcOut0Shape));
    offset += sizeOfShape(biasFcOut0Shape) * Float32Array.BYTES_PER_ELEMENT;
    const weight216Data = new Float32Array(
        buffer, offset, sizeOfShape(weight216Shape));
    offset += sizeOfShape(weight216Shape) * Float32Array.BYTES_PER_ELEMENT;
    const biasFcOut2Data = new Float32Array(
        buffer, offset, sizeOfShape(biasFcOut2Shape));
    offset += sizeOfShape(biasFcOut2Shape) * Float32Array.BYTES_PER_ELEMENT;
    const weight217Data = new Float32Array(
        buffer, offset, sizeOfShape(weight217Shape));
    offset += sizeOfShape(weight217Shape) * Float32Array.BYTES_PER_ELEMENT;
    const biasFcOut4Data = new Float32Array(
        buffer, offset, sizeOfShape(biasFcOut4Shape));
    offset += sizeOfShape(biasFcOut4Shape) * Float32Array.BYTES_PER_ELEMENT;

    // Create constants
    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();
    const weight172 = builder.constant(
        {type: 'float32', dimensions: weight172Shape}, weight172Data);
    const biasFcIn0 = builder.constant(
        {type: 'float32', dimensions: biasFcIn0Shape}, biasFcIn0Data);
    const weight192 = builder.constant(
        {type: 'float32', dimensions: weight192Shape}, weight192Data);
    const recurrentWeight193 = builder.constant(
        {type: 'float32', dimensions: recurrentWeight193Shape},
        recurrentWeight193Data);
    const bias194 = builder.constant(
        {type: 'float32', dimensions: bias194Shape}, bias194Data);
    const recurrentBias194 = builder.constant(
        {type: 'float32', dimensions: recurrentBias194Shape},
        recurrentBias194Data);
    const weight212 = builder.constant(
        {type: 'float32', dimensions: weight212Shape}, weight212Data);
    const recurrentWeight213 = builder.constant(
        {type: 'float32', dimensions: recurrentWeight213Shape},
        recurrentWeight213Data);
    const bias214 = builder.constant(
        {type: 'float32', dimensions: bias214Shape}, bias214Data);
    const recurrentBias214 = builder.constant(
        {type: 'float32', dimensions: recurrentBias214Shape},
        recurrentBias214Data);
    const weight215 = builder.constant(
        {type: 'float32', dimensions: weight215Shape}, weight215Data);
    const biasFcOut0 = builder.constant(
        {type: 'float32', dimensions: biasFcOut0Shape}, biasFcOut0Data);
    const weight216 = builder.constant(
        {type: 'float32', dimensions: weight216Shape}, weight216Data);
    const biasFcOut2 = builder.constant(
        {type: 'float32', dimensions: biasFcOut2Shape}, biasFcOut2Data);
    const weight217 = builder.constant(
        {type: 'float32', dimensions: weight217Shape}, weight217Data);
    const biasFcOut4 = builder.constant(
        {type: 'float32', dimensions: biasFcOut4Shape}, biasFcOut4Data);

    // Build up the network
    const inputShape = [batchSize, frames, 161];
    const input = builder.input(
        'input', {type: 'float32', dimensions: inputShape});
    const matmul18 = builder.matmul(input, weight172);
    const add19 = builder.add(matmul18, biasFcIn0);
    const relu20 = builder.relu(add19);
    const transpose31 = builder.transpose(relu20, {permutation: [1, 0, 2]});
    const [, gru93] = builder.gru(
        transpose31, weight192, recurrentWeight193, frames, hiddenSize,
        {bias: bias194, recurrentBias: recurrentBias194, returnSequence: true});
    const squeeze95 = builder.squeeze(gru93, {axes: [1]});
    const [, gru156] = builder.gru(
        squeeze95, weight212, recurrentWeight213, frames, hiddenSize,
        {bias: bias214, recurrentBias: recurrentBias214, returnSequence: true});
    const squeeze158 = builder.squeeze(gru156, {axes: [1]});
    const transpose159 = builder.transpose(
        squeeze158, {permutation: [1, 0, 2]});
    const matmul161 = builder.matmul(transpose159, weight215);
    const add162 = builder.add(matmul161, biasFcOut0);
    const relu163 = builder.relu(add162);
    const matmul165 = builder.matmul(relu163, weight216);
    const add166 = builder.add(matmul165, biasFcOut2);
    const relu167 = builder.relu(add166);
    const matmul169 = builder.matmul(relu167, weight217);
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
