
'use strict';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

async function buildConstantByNpy(builder, url) {
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
  const response = await fetch(url);
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
  return builder.constant({type, dimensions}, typedArray);
}

/* eslint max-len: ["error", { "code": 130 }] */

// Noise Suppression Net 2 (NSNet2) Baseline Model for Deep Noise Suppression Challenge (DNS) 2021.
export class NSNet2 {
  constructor() {
    this.model = null;
    this.compiledModel = null;
    this.frameSize = 161;
    this.hiddenSize = 400;
  }

  async load(baseUrl, batchSize, frames) {
    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();
    // Create constants by loading pre-trained data from .npy files.
    const weight172 = await buildConstantByNpy(builder, baseUrl + '172.npy');
    const biasFcIn0 = await buildConstantByNpy(builder, baseUrl + 'fc_in_0_bias.npy');
    const weight192 = await buildConstantByNpy(builder, baseUrl + '192.npy');
    const recurrentWeight193 = await buildConstantByNpy(builder, baseUrl + '193.npy');
    const bias194 = await buildConstantByNpy(builder, baseUrl + '194_0.npy');
    const recurrentBias194 = await buildConstantByNpy(builder, baseUrl + '194_1.npy');
    const weight212 = await buildConstantByNpy(builder, baseUrl + '212.npy');
    const recurrentWeight213 = await buildConstantByNpy(builder, baseUrl + '213.npy');
    const bias214 = await buildConstantByNpy(builder, baseUrl + '214_0.npy');
    const recurrentBias214 = await buildConstantByNpy(builder, baseUrl + '214_1.npy');
    const weight215 = await buildConstantByNpy(builder, baseUrl + '215.npy');
    const biasFcOut0 = await buildConstantByNpy(builder, baseUrl + 'fc_out_0_bias.npy');
    const weight216 = await buildConstantByNpy(builder, baseUrl + '216.npy');
    const biasFcOut2 = await buildConstantByNpy(builder, baseUrl + 'fc_out_2_bias.npy');
    const weight217 = await buildConstantByNpy(builder, baseUrl + '217.npy');
    const biasFcOut4 = await buildConstantByNpy(builder, baseUrl + 'fc_out_4_bias.npy');
    // Build up the network.
    const input = builder.input('input', {type: 'float32', dimensions: [batchSize, frames, this.frameSize]});
    const relu20 = builder.relu(builder.add(builder.matmul(input, weight172), biasFcIn0));
    const transpose31 = builder.transpose(relu20, {permutation: [1, 0, 2]});
    const initialState92 = builder.input('initialState92', {type: 'float32', dimensions: [1, batchSize, this.hiddenSize]});
    const [gru94, gru93] = builder.gru(transpose31, weight192, recurrentWeight193, frames, this.hiddenSize,
        {bias: bias194, recurrentBias: recurrentBias194, initialHiddenState: initialState92, returnSequence: true});
    const squeeze95 = builder.squeeze(gru93, {axes: [1]});
    const initialState155 = builder.input('initialState155', {type: 'float32', dimensions: [1, batchSize, this.hiddenSize]});
    const [gru157, gru156] = builder.gru(squeeze95, weight212, recurrentWeight213, frames, this.hiddenSize,
        {bias: bias214, recurrentBias: recurrentBias214, initialHiddenState: initialState155, returnSequence: true});
    const squeeze158 = builder.squeeze(gru156, {axes: [1]});
    const transpose159 = builder.transpose(squeeze158, {permutation: [1, 0, 2]});
    const relu163 = builder.relu(builder.add(builder.matmul(transpose159, weight215), biasFcOut0));
    const relu167 = builder.relu(builder.add(builder.matmul(relu163, weight216), biasFcOut2));
    const output = builder.sigmoid(builder.add(builder.matmul(relu167, weight217), biasFcOut4));
    this.model = builder.createModel({output, gru94, gru157});
  }

  async compile(options) {
    this.compiledModel = await this.model.compile(options);
  }

  async compute(inputBuffer, initialState92Buffer, initialState155Buffer) {
    const inputs = {
      input: {buffer: inputBuffer},
      initialState92: {buffer: initialState92Buffer},
      initialState155: {buffer: initialState155Buffer},
    };
    return await this.compiledModel.compute(inputs);
  }
}
