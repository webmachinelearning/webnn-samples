'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

// SqueezeNet 1.0 model with 'nhwc' layout
export class SqueezeNetNhwc {
  constructor(dataType = 'float32') {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.targetDataType_ = dataType;
    this.weightsUrl_ = weightsOrigin() +
    '/test-data/models/squeezenet1.0_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: 'nhwc',
      labelUrl: './labels/labels1001.txt',
      inputShape: [1, 224, 224, 3],
    };
    this.outputShape_ = [1, 1001];
  }

  async buildConv_(input, name, options = {}) {
    const prefix = this.weightsUrl_ + name;
    const weightsName = prefix + '_kernel.npy';
    const weights = await buildConstantByNpy(
        this.builder_, weightsName, this.targetDataType_);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = buildConstantByNpy(
        this.builder_, biasName, this.targetDataType_);
    options.inputLayout = 'nhwc';
    options.filterLayout = 'ohwi';
    options.bias = await bias;
    // WebNN spec drops autoPad support, compute the explicit padding instead.
    if (options.autoPad == 'same-upper') {
      const isShapeMethod = typeof weights.shape === 'function';
      const inputShape = isShapeMethod ? (await input).shape() :
          (await input).shape;
      const weightsShape = isShapeMethod ? weights.shape() : weights.shape;
      options.padding =
        computePadding2DForAutoPad(
            /* nwhc */[inputShape[1], inputShape[2]],
            /* ohwi */[weightsShape[1], weightsShape[2]],
            options.strides, options.dilations, options.autoPad);
    }
    const conv2d = this.builder_.conv2d(await input, weights, options);
    return this.builder_.relu(conv2d);
  }

  async buildFire_(input, name) {
    const convSqueeze = this.buildConv_(input, name + '_squeeze');
    const convE1x1 = this.buildConv_(convSqueeze, name + '_e1x1');
    const convE3x3 = this.buildConv_(
        convSqueeze, name + '_e3x3', {padding: [1, 1, 1, 1]});
    return this.builder_.concat([await convE1x1, await convE3x3], 3);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const strides = [2, 2];
    const layout = 'nhwc';
    const inputDesc = {
      dataType: 'float32',
      dimensions: this.inputOptions.inputShape,
      shape: this.inputOptions.inputShape,
    };
    let placeholder = this.builder_.input('input', inputDesc);
    inputDesc.usage = MLTensorUsage.WRITE;
    inputDesc.writable = true;
    this.inputTensor_ = await this.context_.createTensor(inputDesc);
    this.outputTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: this.outputShape_,
      shape: this.outputShape_,
      usage: MLTensorUsage.READ,
      readable: true,
    });

    if (this.targetDataType_ === 'float16') {
      placeholder = this.builder_.cast(placeholder, 'float16');
    }
    const conv1 = this.buildConv_(
        placeholder, 'conv1', {strides, autoPad: 'same-upper'});
    const maxpool1 = this.builder_.maxPool2d(
        await conv1, {windowDimensions: [3, 3], strides, layout});
    const fire2 = this.buildFire_(maxpool1, 'fire2');
    const fire3 = this.buildFire_(fire2, 'fire3');
    const fire4 = this.buildFire_(fire3, 'fire4');
    const maxpool4 = this.builder_.maxPool2d(
        await fire4, {windowDimensions: [3, 3], strides, layout});
    const fire5 = this.buildFire_(maxpool4, 'fire5');
    const fire6 = this.buildFire_(fire5, 'fire6');
    const fire7 = this.buildFire_(fire6, 'fire7');
    const fire8 = this.buildFire_(fire7, 'fire8');
    const maxpool8 = this.builder_.maxPool2d(
        await fire8, {windowDimensions: [3, 3], strides, layout});
    const fire9 = this.buildFire_(maxpool8, 'fire9');
    const conv10 = this.buildConv_(fire9, 'conv10');
    const averagePool2d = this.builder_.averagePool2d(
        await conv10, {windowDimensions: [13, 13], layout});
    const reshape = this.builder_.reshape(averagePool2d, [1, 1001]);
    const softmax = this.builder_.softmax(reshape, 1);

    if (this.targetDataType_ === 'float16') {
      return this.builder_.cast(softmax, 'float32');
    }
    return softmax;
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  async compute(inputBuffer) {
    this.context_.writeTensor(this.inputTensor_, inputBuffer);
    const inputs = {'input': this.inputTensor_};
    const outputs = {'output': this.outputTensor_};
    this.context_.dispatch(this.graph_, inputs, outputs);
    const results = await this.context_.readTensor(this.outputTensor_);
    return new Float32Array(results);
  }
}
