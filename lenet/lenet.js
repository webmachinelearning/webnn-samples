
'use strict';

import {getBufferFromUrl, sizeOfShape} from '../common/utils.js';

export class LeNet {
  constructor(url) {
    this.url_ = url;
    this.graph_ = null;
    this.builder_ = null;
  }

  async load(devicePreference) {
    const arrayBuffer = await getBufferFromUrl(this.url_);
    const WEIGHTS_FILE_SIZE = 1724336;
    if (arrayBuffer.byteLength !== WEIGHTS_FILE_SIZE) {
      throw new Error('Incorrect weights file');
    }

    const context = navigator.ml.createContext({devicePreference});
    this.builder_ = new MLGraphBuilder(context);
    const inputShape = [1, 1, 28, 28];
    const input =
        this.builder_.input('input', {type: 'float32', dimensions: inputShape});

    const conv1FitlerShape = [20, 1, 5, 5];
    let byteOffset = 0;
    const conv1FilterData = new Float32Array(
        arrayBuffer, byteOffset, sizeOfShape(conv1FitlerShape));
    const conv1Filter = this.builder_.constant(
        {type: 'float32', dimensions: conv1FitlerShape},
        conv1FilterData);
    byteOffset +=
        sizeOfShape(conv1FitlerShape) * Float32Array.BYTES_PER_ELEMENT;
    const conv1 = this.builder_.conv2d(input, conv1Filter);

    const add1BiasShape = [1, 20, 1, 1];
    const add1BiasData =
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add1BiasShape));
    const add1Bias = this.builder_.constant(
        {type: 'float32', dimensions: add1BiasShape}, add1BiasData);
    byteOffset += sizeOfShape(add1BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add1 = this.builder_.add(conv1, add1Bias);

    const pool1WindowShape = [2, 2];
    const pool1Strides = [2, 2];
    const pool1 =
        this.builder_.maxPool2d(add1, {windowDimensions: pool1WindowShape,
          strides: pool1Strides});

    const conv2FilterShape = [50, 20, 5, 5];
    const conv2Filter = this.builder_.constant(
        {type: 'float32', dimensions: conv2FilterShape},
        new Float32Array(
            arrayBuffer, byteOffset, sizeOfShape(conv2FilterShape)));
    byteOffset +=
        sizeOfShape(conv2FilterShape) * Float32Array.BYTES_PER_ELEMENT;
    const conv2 = this.builder_.conv2d(pool1, conv2Filter);

    const add2BiasShape = [1, 50, 1, 1];
    const add2Bias = this.builder_.constant(
        {type: 'float32', dimensions: add2BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add2BiasShape)));
    byteOffset += sizeOfShape(add2BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add2 = this.builder_.add(conv2, add2Bias);

    const pool2WindowShape = [2, 2];
    const pool2Strides = [2, 2];
    const pool2 =
        this.builder_.maxPool2d(add2, {windowDimensions: pool2WindowShape,
          strides: pool2Strides});

    const reshape1Shape = [1, -1];
    const reshape1 = this.builder_.reshape(pool2, reshape1Shape);

    // skip the new shape, 2 int64 values
    byteOffset += 2 * 8;

    const matmul1Shape = [500, 800];
    const matmul1Weights = this.builder_.constant(
        {type: 'float32', dimensions: matmul1Shape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul1Shape)));
    byteOffset += sizeOfShape(matmul1Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul1WeightsTransposed = this.builder_.transpose(matmul1Weights);
    const matmul1 = this.builder_.matmul(reshape1, matmul1WeightsTransposed);

    const add3BiasShape = [1, 500];
    const add3Bias = this.builder_.constant(
        {type: 'float32', dimensions: add3BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add3BiasShape)));
    byteOffset += sizeOfShape(add3BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add3 = this.builder_.add(matmul1, add3Bias);

    const relu = this.builder_.relu(add3);

    const reshape2Shape = [1, -1];
    const reshape2 = this.builder_.reshape(relu, reshape2Shape);

    const matmul2Shape = [10, 500];
    const matmul2Weights = this.builder_.constant(
        {type: 'float32', dimensions: matmul2Shape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul2Shape)));
    byteOffset += sizeOfShape(matmul2Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul2WeightsTransposed = this.builder_.transpose(matmul2Weights);
    const matmul2 = this.builder_.matmul(reshape2, matmul2WeightsTransposed);

    const add4BiasShape = [1, 10];
    const add4Bias = this.builder_.constant(
        {type: 'float32', dimensions: add4BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add4BiasShape)));
    const add4 = this.builder_.add(matmul2, add4Bias);

    return this.builder_.softmax(add4);
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
  }

  predict(inputBuffer, outputBuffer) {
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    this.graph_.compute(inputs, outputs);
  }
}
