
'use strict';

const nn = navigator.ml.getNeuralNetworkContext();

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

export class LeNet {
  constructor(url) {
    this.url_ = url;
    this.model_ = null;
    this.compilation_ = null;
  }

  async load() {
    const response = await fetch(this.url_);
    const arrayBuffer = await response.arrayBuffer();
    const WEIGHTS_FILE_SIZE = 1724336;
    if (arrayBuffer.byteLength !== WEIGHTS_FILE_SIZE) {
      throw new Error('Incorrect weights file');
    }

    const builder = nn.createModelBuilder();
    const inputShape = [1, 1, 28, 28];
    const input =
        builder.input('input', {type: 'float32', dimensions: inputShape});

    const conv1FitlerShape = [20, 1, 5, 5];
    let byteOffset = 0;
    const conv1FilterData = new Float32Array(
        arrayBuffer, byteOffset, sizeOfShape(conv1FitlerShape));
    const conv1Filter = builder.constant(
        {type: 'float32', dimensions: conv1FitlerShape},
        conv1FilterData);
    byteOffset +=
        sizeOfShape(conv1FitlerShape) * Float32Array.BYTES_PER_ELEMENT;
    const conv1 = builder.conv2d(input, conv1Filter);

    const add1BiasShape = [1, 20, 1, 1];
    const add1BiasData =
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add1BiasShape));
    const add1Bias = builder.constant(
        {type: 'float32', dimensions: add1BiasShape}, add1BiasData);
    byteOffset += sizeOfShape(add1BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add1 = builder.add(conv1, add1Bias);

    const pool1WindowShape = [2, 2];
    const pool1Strides = [2, 2];
    const pool1 =
        builder.maxPool2d(add1, {windowDimensions: pool1WindowShape,
          strides: pool1Strides});

    const conv2FilterShape = [50, 20, 5, 5];
    const conv2Filter = builder.constant(
        {type: 'float32', dimensions: conv2FilterShape},
        new Float32Array(
            arrayBuffer, byteOffset, sizeOfShape(conv2FilterShape)));
    byteOffset +=
        sizeOfShape(conv2FilterShape) * Float32Array.BYTES_PER_ELEMENT;
    const conv2 = builder.conv2d(pool1, conv2Filter);

    const add2BiasShape = [1, 50, 1, 1];
    const add2Bias = builder.constant(
        {type: 'float32', dimensions: add2BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add2BiasShape)));
    byteOffset += sizeOfShape(add2BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add2 = builder.add(conv2, add2Bias);

    const pool2WindowShape = [2, 2];
    const pool2Strides = [2, 2];
    const pool2 =
        builder.maxPool2d(add2, {windowDimensions: pool2WindowShape,
          strides: pool2Strides});

    const reshape1Shape = [1, -1];
    const reshape1 = builder.reshape(pool2, reshape1Shape);

    // skip the new shape, 2 int64 values
    byteOffset += 2 * 8;

    const matmul1Shape = [500, 800];
    const matmul1Weights = builder.constant(
        {type: 'float32', dimensions: matmul1Shape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul1Shape)));
    byteOffset += sizeOfShape(matmul1Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul1WeightsTransposed = builder.transpose(matmul1Weights);
    const matmul1 = builder.matmul(reshape1, matmul1WeightsTransposed);

    const add3BiasShape = [1, 500];
    const add3Bias = builder.constant(
        {type: 'float32', dimensions: add3BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add3BiasShape)));
    byteOffset += sizeOfShape(add3BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add3 = builder.add(matmul1, add3Bias);

    const relu = builder.relu(add3);

    const reshape2Shape = [1, -1];
    const reshape2 = builder.reshape(relu, reshape2Shape);

    const matmul2Shape = [10, 500];
    const matmul2Weights = builder.constant(
        {type: 'float32', dimensions: matmul2Shape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul2Shape)));
    byteOffset += sizeOfShape(matmul2Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul2WeightsTransposed = builder.transpose(matmul2Weights);
    const matmul2 = builder.matmul(reshape2, matmul2WeightsTransposed);

    const add4BiasShape = [1, 10];
    const add4Bias = builder.constant(
        {type: 'float32', dimensions: add4BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add4BiasShape)));
    const add4 = builder.add(matmul2, add4Bias);

    const softmax = builder.softmax(add4);

    this.model_ = builder.createModel({'output': softmax});
  }

  async compile(options) {
    this.compilation_ = await this.model_.compile(options);
  }

  async predict(inputBuffer) {
    const inputs = {input: {buffer: inputBuffer}};
    const outputs = await this.compilation_.compute(inputs);
    return outputs.output.buffer;
  }
}
