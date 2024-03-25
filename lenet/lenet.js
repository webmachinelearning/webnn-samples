
'use strict';

import {getBufferFromUrl, sizeOfShape, permuteData} from '../common/utils.js';

export class LeNet {
  constructor(url, layout) {
    this.context_ = null;
    this.url_ = url;
    this.graph_ = null;
    this.builder_ = null;
    this.layout_ = layout;
    this.nchwToNhwcPermutation_ = [0, 2, 3, 1];
    this.nhwcToNchwPermutation_ = [0, 3, 1, 2];
    this.oihwToOhwiPermutation_ = [0, 2, 3, 1];
  }

  async load(contextOptions) {
    const arrayBuffer = await getBufferFromUrl(this.url_);
    const WEIGHTS_FILE_SIZE = 1724336;
    if (arrayBuffer.byteLength !== WEIGHTS_FILE_SIZE) {
      throw new Error('Incorrect weights file');
    }

    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const inputShape = /* nchw */ [1, 1, 28, 28];
    let input = this.builder_.input('input', {
      dataType: 'float32',
      dimensions: inputShape,
    });

    if (this.layout_ === 'nhwc') {
      input = this.builder_.transpose(
          input, {permutation: this.nchwToNhwcPermutation_});
    }

    const conv1Options = {};
    if (this.layout_ === 'nhwc') {
      conv1Options.inputLayout = 'nhwc';
      conv1Options.filterLayout = 'ohwi';
    }
    let conv1FitlerShape = /* oihw */ [20, 1, 5, 5];
    let byteOffset = 0;
    let conv1FilterData = new Float32Array(
        arrayBuffer, byteOffset, sizeOfShape(conv1FitlerShape));
    if (this.layout_ === 'nhwc') {
      [conv1FilterData, conv1FitlerShape] =
          permuteData(
              conv1FilterData, conv1FitlerShape, this.oihwToOhwiPermutation_);
    }
    const conv1Filter = this.builder_.constant(
        {dataType: 'float32', dimensions: conv1FitlerShape},
        conv1FilterData);
    byteOffset +=
        sizeOfShape(conv1FitlerShape) * Float32Array.BYTES_PER_ELEMENT;

    const add1BiasShape = [20];
    const add1BiasData =
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add1BiasShape));
    const add1Bias = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: add1BiasShape},
        add1BiasData,
    );
    byteOffset += sizeOfShape(add1BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    conv1Options.bias = add1Bias;

    const conv1 = this.builder_.conv2d(input, conv1Filter, conv1Options);

    const pool1WindowShape = [2, 2];
    const pool1Strides = [2, 2];
    const pool1 =
        this.builder_.maxPool2d(conv1, {windowDimensions: pool1WindowShape,
          strides: pool1Strides, layout: this.layout_});

    const conv2Options = {};
    if (this.layout_ === 'nhwc') {
      conv2Options.inputLayout = 'nhwc';
      conv2Options.filterLayout = 'ohwi';
    }
    let conv2FilterShape = /* oihw */ [50, 20, 5, 5];
    let conv2FilterData = new Float32Array(
        arrayBuffer, byteOffset, sizeOfShape(conv2FilterShape));
    if (this.layout_ === 'nhwc') {
      [conv2FilterData, conv2FilterShape] =
          permuteData(
              conv2FilterData, conv2FilterShape, this.oihwToOhwiPermutation_);
    }
    const conv2Filter = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: conv2FilterShape},
        conv2FilterData);
    byteOffset +=
        sizeOfShape(conv2FilterShape) * Float32Array.BYTES_PER_ELEMENT;

    const add2BiasShape = [50];
    const add2Bias = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: add2BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add2BiasShape)));
    byteOffset += sizeOfShape(add2BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    conv2Options.bias = add2Bias;

    const conv2 = this.builder_.conv2d(pool1, conv2Filter, conv2Options);

    const pool2WindowShape = [2, 2];
    const pool2Strides = [2, 2];
    let pool2 =
        this.builder_.maxPool2d(conv2, {windowDimensions: pool2WindowShape,
          strides: pool2Strides, layout: this.layout_});

    if (this.layout_ === 'nhwc') {
      pool2 = this.builder_.transpose(
          pool2, {permutation: this.nhwcToNchwPermutation_});
    }

    const reshape1Shape = [1, 800];
    const reshape1 = this.builder_.reshape(pool2, reshape1Shape);

    // skip the new shape, 2 int64 values
    byteOffset += 2 * 8;

    const matmul1Shape = [500, 800];
    const matmul1Weights = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: matmul1Shape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul1Shape)));
    byteOffset += sizeOfShape(matmul1Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul1WeightsTransposed = this.builder_.transpose(matmul1Weights);
    const matmul1 = this.builder_.gemm(reshape1, matmul1WeightsTransposed);

    const add3BiasShape = [1, 500];
    const add3Bias = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: add3BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add3BiasShape)));
    byteOffset += sizeOfShape(add3BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add3 = this.builder_.add(matmul1, add3Bias);

    const relu = this.builder_.relu(add3);

    const reshape2Shape = [1, 500];
    const reshape2 = this.builder_.reshape(relu, reshape2Shape);

    const matmul2Shape = [10, 500];
    const matmul2Weights = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: matmul2Shape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul2Shape)));
    byteOffset += sizeOfShape(matmul2Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul2WeightsTransposed = this.builder_.transpose(matmul2Weights);
    const matmul2 = this.builder_.gemm(reshape2, matmul2WeightsTransposed);

    const add4BiasShape = [1, 10];
    const add4Bias = this.builder_.constant(
        {type: 'float32', dataType: 'float32', dimensions: add4BiasShape},
        new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add4BiasShape)));
    const add4 = this.builder_.add(matmul2, add4Bias);

    return this.builder_.softmax(add4);
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  async compute(inputBuffer, outputBuffer) {
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results;
  }
}
