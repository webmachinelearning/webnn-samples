'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

const autoPad = 'same-upper';
const strides = [2, 2];
const layout = 'nhwc';

// ResNet 50 V2 model with 'nhwc' layout
export class ResNet50V2Nhwc {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/resnet50v2_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: layout,
      labelUrl: './labels/labels1001.txt',
      inputShape: [1, 224, 224, 3],
    };
    this.outputShape_ = [1, 1001];
  }

  async buildConv_(input, nameIndices, options = {}, relu = true) {
    let prefix = this.weightsUrl_ + 'resnet_v2_50_';
    // Items in 'nameIndices' represent the indices of block, unit, conv
    // respectively, except two kinds of specific conv names:
    // 1. contains 'shortcut', e.g.
    // resnet_v2_50_block1_unit_1_bottleneck_v2_shortcut_weights.npy
    // 2. contains 'logits', e.g. resnet_v2_50_logits_weights.npy
    if (nameIndices[0] !== '' && nameIndices[1] !== '') {
      prefix += `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_`;
    }
    if (nameIndices[2] === 'shortcut') {
      prefix += 'shortcut';
    } else if (nameIndices[2] === 'logits') {
      prefix += nameIndices[2];
    } else {
      prefix += 'conv' + nameIndices[2];
    }
    const weightsName = prefix + '_weights.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = buildConstantByNpy(this.builder_, biasName);
    options.inputLayout = layout;
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
    return relu ? this.builder_.relu(conv2d) : conv2d;
  }

  async buildFusedBatchNorm_(input, nameIndices) {
    let prefix = this.weightsUrl_ + 'resnet_v2_50_';
    if (nameIndices[0] === 'postnorm') {
      prefix += 'postnorm';
    } else {
      prefix +=
          `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_preact`;
    }
    const mulParamName = prefix + '_FusedBatchNorm_mul_0_param.npy';
    const mulParam = buildConstantByNpy(this.builder_, mulParamName);
    const addParamName = prefix + '_FusedBatchNorm_add_param.npy';
    const addParam = buildConstantByNpy(this.builder_, addParamName);
    return this.builder_.relu(
        this.builder_.add(
            this.builder_.mul(await input, await mulParam),
            await addParam,
        ),
    );
  }

  async buildBottleneckV2_(
      input, nameIndices, downsample = false, shortcut = true) {
    let residual = await input;

    const fusedBn = this.buildFusedBatchNorm_(await input, nameIndices);
    const conv1 = this.buildConv_(
        await fusedBn, nameIndices.concat(['1']), {autoPad});
    let conv2;
    if (downsample) {
      residual = this.buildConv_(
          await fusedBn, nameIndices.concat(['shortcut']), {autoPad}, false);
    }
    if (!downsample && shortcut) {
      residual = this.builder_.maxPool2d(
          await input, {
            windowDimensions: [2, 2],
            strides,
            layout,
            autoPad,
          },
      );
      conv2 = this.buildConv_(
          await conv1, nameIndices.concat(['2']), {
            strides,
            padding: [1, 1, 1, 1],
          },
      );
    } else {
      conv2 = this.buildConv_(
          await conv1, nameIndices.concat(['2']), {autoPad},
      );
    }
    const conv3 = this.buildConv_(
        await conv2, nameIndices.concat(['3']), {autoPad}, false);
    return this.builder_.add(await conv3, await residual);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const inputDesc = {
      dataType: 'float32',
      dimensions: this.inputOptions.inputShape,
      shape: this.inputOptions.inputShape,
    };
    const input = this.builder_.input('input', inputDesc);
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
    const conv1 = await this.buildConv_(
        input, ['', '', '1'], {strides, padding: [3, 3, 3, 3]}, false);
    const windowDimensions = [3, 3];
    const conv1Shape = typeof conv1.shape === 'function' ?
        conv1.shape() : conv1.shape;
    const pool = this.builder_.maxPool2d(
        conv1, {windowDimensions, strides, layout,
          padding: computePadding2DForAutoPad(
              /* nhwc */ [conv1Shape[1], conv1Shape[2]],
              windowDimensions, strides, /* dilations */ undefined,
              'same-upper')});
    // Block 1
    const bottleneck1 = this.buildBottleneckV2_(pool, ['1', '1'], true);
    const bottleneck2 = this.buildBottleneckV2_(
        bottleneck1, ['1', '2'], false, false);
    const bottleneck3 = this.buildBottleneckV2_(
        bottleneck2, ['1', '3']);

    // Block 2
    const bottleneck4 = this.buildBottleneckV2_(
        bottleneck3, ['2', '1'], true);
    const bottleneck5 = this.buildBottleneckV2_(
        bottleneck4, ['2', '2'], false, false);
    const bottleneck6 = this.buildBottleneckV2_(
        bottleneck5, ['2', '3'], false, false);
    const bottleneck7 = this.buildBottleneckV2_(
        bottleneck6, ['2', '4']);

    // Block 3
    const bottleneck8 = this.buildBottleneckV2_(
        bottleneck7, ['3', '1'], true);
    const loop = async (node, num) => {
      if (num > 5) {
        return node;
      } else {
        const newNode = this.buildBottleneckV2_(
            node, ['3', num.toString()], false, false);
        num++;
        return loop(newNode, num);
      }
    };
    const bottleneck9 = loop(bottleneck8, 2);
    const bottleneck10 = this.buildBottleneckV2_(
        bottleneck9, ['3', '6']);

    // Block 4
    const bottleneck11 = this.buildBottleneckV2_(
        bottleneck10, ['4', '1'], true);
    const bottleneck12 = this.buildBottleneckV2_(
        bottleneck11, ['4', '2'], false, false);
    const bottleneck13 = this.buildBottleneckV2_(
        bottleneck12, ['4', '3'], false, false);

    const fusedBn =
        this.buildFusedBatchNorm_(bottleneck13, ['postnorm']);
    const mean = this.builder_.averagePool2d(await fusedBn, {layout});
    const conv2 = this.buildConv_(
        mean, ['', '', 'logits'], {autoPad}, false);
    const reshape = this.builder_.reshape(await conv2, [1, 1001]);
    return this.builder_.softmax(reshape);
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  // Release the constant tensors of a model
  async compute(inputBuffer) {
    this.context_.writeTensor(this.inputTensor_, inputBuffer);
    const inputs = {'input': this.inputTensor_};
    const outputs = {'output': this.outputTensor_};
    this.context_.dispatch(this.graph_, inputs, outputs);
    const results = await this.context_.readTensor(this.outputTensor_);
    return new Float32Array(results);
  }
}
