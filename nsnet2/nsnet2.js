
'use strict';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

export class NSNet2 {
  constructor() {
    this.baseUrl_ = './';
    this.model_ = null;
    this.compilation_ = null;
    this.hiddenSize = 400;
  }

  async buildConstantByNpy(fileName) {
    const dataTypeMap = new Map([
      ['f2', {type: 'float16', array: Uint16Array}],
      ['f4', {type: 'float32', array: Float32Array}],
      ['f8', {type: 'float64', array: Float64Array}],
      ['i1', {type: 'int8', array: Int8Array}],
      ['i2', {type: 'int16', array: Int16Array}],
      ['i4', {type: 'int32', array: Int32Array}],
      ['i8', {type: 'int64', array: BigInt64Array}],
      ['u1', {type: 'uint8', array: Uint8Array}],
      ['u2', {type: 'uint16', array: Uint16Array}],
      ['u4', {type: 'uint32', array: Uint32Array}],
      ['u8', {type: 'uint64', array: BigUint64Array}],
    ]);
    const response = await fetch(this.baseUrl_ + fileName);
    const buffer = await response.arrayBuffer();
    const npArray = new numpy.Array(new Uint8Array(buffer));
    if (!dataTypeMap.has(npArray.dataType)) {
      throw new Error(`Data type ${npArray.dataType} is not supported.`);
    }
    const dimensions = npArray.shape;
    const type = dataTypeMap.get(npArray.dataType).type;
    const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
    const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
    const dataView = new DataView(npArray.data.buffer);
    const littleEndian = npArray.byteOrder === '<';
    for (let i = 0; i < sizeOfShape(dimensions); ++i) {
      typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
          i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
    }
    return this.builder.constant({type, dimensions}, typedArray);
  }

  async load(url, batchSize, frames) {
    this.baseUrl_ = url;
    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();
    this.builder = builder;

    // Create constants
    const weight172 = await this.buildConstantByNpy('172.npy');
    const biasFcIn0 = await this.buildConstantByNpy('fc_in_0_bias.npy');
    const weight192 = await this.buildConstantByNpy('192.npy');
    const recurrentWeight193 = await this.buildConstantByNpy('193.npy');
    const data194 = await this.buildConstantByNpy('194.npy');
    const weight212 = await this.buildConstantByNpy('212.npy');
    const recurrentWeight213 = await this.buildConstantByNpy('213.npy');
    const data214 = await this.buildConstantByNpy('214.npy');
    const weight215 = await this.buildConstantByNpy('215.npy');
    const biasFcOut0 = await this.buildConstantByNpy('fc_out_0_bias.npy');
    const weight216 = await this.buildConstantByNpy('216.npy');
    const biasFcOut2 = await this.buildConstantByNpy('fc_out_2_bias.npy');
    const weight217 = await this.buildConstantByNpy('217.npy');
    const biasFcOut4 = await this.buildConstantByNpy('fc_out_4_bias.npy');

    // Build up the network
    const hiddenSize = this.hiddenSize;
    const inputShape = [batchSize, frames, 161];
    const input = builder.input(
        'input', {type: 'float32', dimensions: inputShape});
    const matmul18 = builder.matmul(input, weight172);
    const add19 = builder.add(matmul18, biasFcIn0);
    const relu20 = builder.relu(add19);
    const transpose31 = builder.transpose(relu20, {permutation: [1, 0, 2]});
    const bias194 = builder.slice(data194, [0], [3 * hiddenSize], {axes: [1]});
    const recurrentBias194 = builder.slice(
        data194, [3 * hiddenSize], [-1], {axes: [1]});
    const initialHiddenState92 = builder.input(
        'initialHiddenState92',
        {type: 'float32', dimensions: [1, batchSize, hiddenSize]});
    const [gru94, gru93] = builder.gru(
        transpose31, weight192, recurrentWeight193, frames, hiddenSize,
        {
          bias: bias194, recurrentBias: recurrentBias194,
          initialHiddenState: initialHiddenState92, returnSequence: true,
        });
    const squeeze95 = builder.squeeze(gru93, {axes: [1]});
    const bias214 = builder.slice(data214, [0], [3 * hiddenSize], {axes: [1]});
    const recurrentBias214 = builder.slice(
        data214, [3 * hiddenSize], [-1], {axes: [1]});
    const initialHiddenState155 = builder.input(
        'initialHiddenState155',
        {type: 'float32', dimensions: [1, batchSize, hiddenSize]});
    const [gru157, gru156] = builder.gru(
        squeeze95, weight212, recurrentWeight213, frames, hiddenSize,
        {
          bias: bias214, recurrentBias: recurrentBias214,
          initialHiddenState: initialHiddenState155, returnSequence: true,
        });
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
    this.model_ = builder.createModel({output, gru94, gru157});
  }

  async compile(options) {
    this.compilation_ = await this.model_.compile(options);
  }

  async compute(
      inputBuffer, initialHiddenState92Buffer, initialHiddenState155Buffer) {
    const inputs = {
      input: {buffer: inputBuffer},
      initialHiddenState92: {buffer: initialHiddenState92Buffer},
      initialHiddenState155: {buffer: initialHiddenState155Buffer},
    };
    return await this.compilation_.compute(inputs);
  }

  dispose() {
    this.compilation_.dispose();
  }
}
