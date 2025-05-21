'use strict';

/* eslint max-len: ["error", {"code": 120}] */

// Selfie-Segmenter WebNN model
export class SelfieSegmentationLandscape {
  constructor(deviceType, dataType) {
    this.deviceType_ = deviceType;
    this.dataType_ = dataType;
    this.ArrayType_ = dataType == 'float32' ? Float32Array : Float16Array;
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.outputShape_ = [1, 144, 256, 1];
  }

  async buildConv_(
      input,
      index,
      activation = '',
      options = {},
  ) {
    const weightInfo = this.weightsInfo_[`conv${index}`];
    const weightBuffer = this.weightsBuffer_.slice(
        weightInfo.dataOffset,
        weightInfo.dataOffset + weightInfo.byteLength,
    );

    let weightData = new Float32Array(weightBuffer);
    if (this.dataType_ == 'float16') {
      weightData = this.ArrayType_.from(weightData);
    }
    const weights = this.builder_.constant(
        {shape: weightInfo.shape, dataType: this.dataType_},
        weightData,
    );

    const biasInfo = this.biasesInfo_[`conv${index}`];
    const biasBuffer = this.biasesBuffer_.slice(
        biasInfo.dataOffset,
        biasInfo.dataOffset + biasInfo.byteLength,
    );

    let biasData = new Float32Array(biasBuffer);
    if (this.dataType_ == 'float16') {
      biasData = this.ArrayType_.from(biasData);
    }
    options.bias = this.builder_.constant(
        {shape: biasInfo.shape, dataType: this.dataType_},
        biasData,
    );

    if (this.layout === 'nhwc') {
      const isDepthwise = options.groups > 1 && options.groups == input['shape'][3];
      options.filterLayout = isDepthwise ? 'ihwo' : 'ohwi';
      options.inputLayout = this.layout;
    }

    const conv2d = this.builder_.conv2d(input, weights, options);

    if (activation === 'relu') {
      return this.builder_.relu(conv2d);
    } else if (activation === 'sigmoid') {
      return this.builder_.sigmoid(conv2d);
    } else {
      return conv2d;
    }
  }

  // Subgraph A:
  // input -> Conv -> Add (B: addB_) -> Clip (min: 0, max: 6) -> Mul (A: mulA_) -> Mul (A: Conv)
  //           |                                                                    ^
  //           v                                                                    |
  //           ----------------------------------------------------------------------
  async buildSubGraphA_(input, convIndex, convOptions = {}) {
    const conv = await this.buildConv_(input, convIndex, '', convOptions);
    const add = this.builder_.add(conv, this.addB_);
    const clip = this.builder_.clamp(add, {minValue: 0, maxValue: 6});
    const mul = this.builder_.mul(this.mulA_, clip);
    return this.builder_.mul(conv, mul);
  }

  // Subgraph B: (if optionInput presents, it will be used as input for Mul)
  // input -> GlobalAveragePool -> Conv -> Relu -> Conv -> Sigmoid -> Mul
  //  |                                                        ^
  //  v                                                        |
  //  ----------------------------------------------------------(or optionInput)
  async buildSubGraphB_(input, convIndex, optionInput = undefined) {
    const gAvgPool2d = this.builder_.averagePool2d(await input, {
      layout: this.layout,
    });
    // convIndex
    const conv1 = await this.buildConv_(gAvgPool2d, convIndex, 'relu');
    // convIndex + 1
    const conv2 = await this.buildConv_(conv1, convIndex + 1, 'sigmoid');
    if (optionInput) {
      return this.builder_.mul(optionInput, conv2);
    } else {
      return this.builder_.mul(input, conv2);
    }
  }

  async load() {
    this.context_ = await navigator.ml.createContext({
      deviceType: this.deviceType_,
    });

    // Choose the layout based on the preferred input layout of the context.
    this.layout = this.context_.opSupportLimits().preferredInputLayout;
    this.inputShape =
      this.layout === 'nhwc' ? [1, 144, 256, 3] : [1, 3, 144, 256];

    // Load the weights, bias and info files.
    const weightsResponse = await fetch(
        `./weights/landscape/weights_${this.layout}.bin`,
    );
    this.weightsBuffer_ = await weightsResponse.arrayBuffer();

    const weightInfoResponse = await fetch(
        `./weights/landscape/weights_${this.layout}.json`,
    );
    this.weightsInfo_ = await weightInfoResponse.json();

    // Different layouts have the same bias
    const biasResponse = await fetch(`./weights/landscape/biases.bin`);
    this.biasesBuffer_ = await biasResponse.arrayBuffer();
    this.biasInfoResponse_ = await fetch(`./weights/landscape/biases.json`);
    this.biasesInfo_ = await this.biasInfoResponse_.json();

    this.builder_ = new MLGraphBuilder(this.context_);
    const strides = [2, 2];

    const inputDesc = {
      dataType: this.dataType_,
      shape: this.inputShape,
    };
    const input = this.builder_.input('input', inputDesc);
    inputDesc.writable = true;
    this.inputTensor_ = await this.context_.createTensor(inputDesc);
    this.outputTensor_ = await this.context_.createTensor({
      dataType: this.dataType_,
      shape: this.outputShape_,
      readable: true,
    });

    this.addB_ = this.builder_.constant(
        {dataType: this.dataType_, shape: [1, 1, 1, 1]},
        new this.ArrayType_([3]),
    );
    this.mulA_ = this.builder_.constant(
        {dataType: this.dataType_, shape: []},
        new this.ArrayType_([0.1666666716337204]),
    );

    // name: mul_1 (contains conv0) Conv__158
    const subGraphA0 = await this.buildSubGraphA_(input, 0, {
      strides,
      padding: [0, 1, 0, 1],
    });

    // Conv__161
    const conv1 = await this.buildConv_(subGraphA0, 1, 'relu');
    // Conv__162
    const conv2 = await this.buildConv_(
        conv1,
        2,
        'relu',
        {
          strides,
          padding: [0, 1, 0, 1],
          groups: 16,
        },
    );

    // name: multiply, Conv__165, Conv__166 (contains conv3, conv4)
    const subGraphB0 = await this.buildSubGraphB_(conv2, 3);

    // name: Conv__167
    const conv5 = await this.buildConv_(subGraphB0, 5, '');
    // name: Conv__171
    const conv6 = await this.buildConv_(conv5, 6, 'relu');
    // name: Conv__172
    const conv7 = await this.buildConv_(
        conv6,
        7,
        'relu',
        {
          strides,
          padding: [0, 1, 0, 1],
          groups: 72,
        },
    );
    // name: Conv__173
    const conv8 = await this.buildConv_(conv7, 8, '');
    // name: Conv__176
    const conv9 = await this.buildConv_(conv8, 9, 'relu');
    // name: Conv__177
    const conv10 = await this.buildConv_(
        conv9,
        10,
        'relu',
        {
          padding: [1, 1, 1, 1],
          groups: 88,
        },
    );
    // name: Conv__178
    const conv11 = await this.buildConv_(conv10, 11, '');

    // name: add__xeno_compat__1
    const add0 = this.builder_.add(conv11, conv8);

    // Conv__183
    const subGraphA1 = await this.buildSubGraphA_(add0, 12);

    // Conv__186
    const subGraphA2 = await this.buildSubGraphA_(
        subGraphA1,
        13,
        {
          strides,
          padding: [1, 2, 1, 2],
          groups: 96,
        },
    );

    // Conv__189, Conv__190 (contains: conv14, conv15)
    const subGraphB1 = await this.buildSubGraphB_(subGraphA2, 14);

    // Conv__191
    const conv15 = await this.buildConv_(subGraphB1, 16, '');

    // Conv__194
    const subGraphA3 = await this.buildSubGraphA_(conv15, 17);

    // Conv__197
    const subGraphA4 = await this.buildSubGraphA_(
        subGraphA3,
        18,
        {
          padding: [2, 2, 2, 2],
          groups: 128,
        },
    );

    // Conv__200, Conv__201 (contains: conv19, conv20)
    const subGraphB2 = await this.buildSubGraphB_(subGraphA4, 19);

    // Conv__202
    const conv21 = await this.buildConv_(subGraphB2, 21, '');

    // name: add_1__xeno_compat__1
    const add1 = this.builder_.add(conv21, conv15);

    // Conv__205
    const subGraphA5 = await this.buildSubGraphA_(add1, 22);

    // Conv__208
    const subGraphA6 = await this.buildSubGraphA_(
        subGraphA5,
        23,
        {
          padding: [2, 2, 2, 2],
          groups: 128,
        },
    );

    // Conv__211, Conv__212 (contains: conv24, conv25)
    const subGraphB3 = await this.buildSubGraphB_(subGraphA6, 24);

    // Conv__213
    const conv26 = await this.buildConv_(subGraphB3, 26, '');

    // name: add_2__xeno_compat__1
    const add2 = this.builder_.add(conv26, add1);

    // Conv__216
    const subGraphA7 = await this.buildSubGraphA_(add2, 27);
    // Conv__219
    const subGraphA8 = await this.buildSubGraphA_(
        subGraphA7,
        28,
        {
          padding: [2, 2, 2, 2],
          groups: 96,
        },
    );

    // Conv__222, Conv__223 (contains: conv29, conv30)
    const subGraphB4 = await this.buildSubGraphB_(subGraphA8, 29);

    // Conv__224
    const conv31 = await this.buildConv_(subGraphB4, 31, '');

    // name: add_3__xeno_compat__1
    const add3 = this.builder_.add(conv31, add2);

    // Conv__227
    const subGraphA9 = await this.buildSubGraphA_(add3, 32);
    // Conv__230
    const subGraphA10 = await this.buildSubGraphA_(
        subGraphA9,
        33,
        {
          padding: [2, 2, 2, 2],
          groups: 96,
        },
    );

    // Conv__233, Conv__234 (contains: conv34, conv35)
    const subGraphB5 = await this.buildSubGraphB_(subGraphA10, 34);

    // Conv__235
    const conv36 = await this.buildConv_(subGraphB5, 36, '');

    // name: add_4__xeno_compat__1
    const add4 = this.builder_.add(conv36, add3);

    // Conv__239
    const conv37 = await this.buildConv_(add4, 37, 'relu');

    // name: global_average_pooling2d_6
    const gAvgPool2d0 = this.builder_.averagePool2d(add4, {
      layout: this.layout,
    });

    // Conv__238
    const conv38 = await this.buildConv_(gAvgPool2d0, 38, 'sigmoid');

    // name: multiply_6
    const mul0 = this.builder_.mul(conv37, conv38);

    // Resize 0
    const resample0 = this.builder_.resample2d(mul0, {
      scales: [2, 2],
      mode: 'linear',
      axes: this.layout === 'nhwc' ? [1, 2] : [2, 3],
    });

    // Conv__240
    const conv39 = await this.buildConv_(resample0, 39, '');

    // name: add_5__xeno_compat__1
    const add5 = this.builder_.add(conv39, add0);

    // Conv__243, Conv__244 (contains: conv40, conv41)
    const subGraphB6 = await this.buildSubGraphB_(add5, 40, add0);

    // name: add_6__xeno_compat__1
    const add6 = this.builder_.add(subGraphB6, conv39);

    // Conv__245
    const conv42 = await this.buildConv_(add6, 42, 'relu');
    // Conv__248
    const conv43 = await this.buildConv_(
        conv42,
        43,
        'relu',
        {
          padding: [1, 1, 1, 1],
          groups: 24,
        },
    );

    // name: add_7__xeno_compat__1
    const add7 = this.builder_.add(conv42, conv43);

    // Resize 1
    const resample1 = this.builder_.resample2d(add7, {
      scales: [2, 2],
      mode: 'linear',
      axes: this.layout === 'nhwc' ? [1, 2] : [2, 3],
    });

    // Conv__249
    const conv44 = await this.buildConv_(resample1, 44, '');

    // name: add_8__xeno_compat__1
    const add8 = this.builder_.add(conv5, conv44);

    // Conv__252, Conv__253 (contains: conv45, conv46)
    const subGraphB7 = await this.buildSubGraphB_(add8, 45, conv5);

    // name: add_9__xeno_compat__1
    const add9 = this.builder_.add(subGraphB7, conv44);

    // Conv__254
    const conv47 = await this.buildConv_(add9, 47, 'relu');
    // Conv__257
    const conv48 = await this.buildConv_(
        conv47,
        48,
        'relu',
        {
          padding: [1, 1, 1, 1],
          groups: 16,
        },
    );

    // name: add_10__xeno_compat__1
    const add10 = this.builder_.add(conv47, conv48);

    // Resize 2
    const resample2 = this.builder_.resample2d(add10, {
      scales: [2, 2],
      mode: 'linear',
      axes: this.layout === 'nhwc' ? [1, 2] : [2, 3],
    });

    // Conv__258
    const conv49 = await this.buildConv_(resample2, 49, '');

    // name: add_11__xeno_compat__1
    const add11 = this.builder_.add(subGraphA0, conv49);

    // Conv__261, Conv__262 (contains: conv50, conv51)
    const subGraphB8 = await this.buildSubGraphB_(add11, 50, subGraphA0);

    // name: add_12__xeno_compat__1
    const add12 = this.builder_.add(subGraphB8, conv49);

    // Conv__263
    const conv52 = await this.buildConv_(add12, 52, 'relu');
    // Conv__266
    const conv53 = await this.buildConv_(
        conv52,
        53,
        'relu',
        {
          padding: [1, 1, 1, 1],
          groups: 16,
        },
    );

    // name: add_13__xeno_compat__1
    const add13 = this.builder_.add(conv52, conv53);

    // ConvTranspose
    const convTransposeWInfo = this.weightsInfo_['convTranspose0'];
    const convTransposeWBuffer = this.weightsBuffer_.slice(
        convTransposeWInfo.dataOffset,
        convTransposeWInfo.dataOffset + convTransposeWInfo.byteLength,
    );

    let convTransposeWData = new Float32Array(convTransposeWBuffer);
    if (this.dataType_ == 'float16') {
      convTransposeWData = this.ArrayType_.from(convTransposeWData);
    }
    const convTransposeW = this.builder_.constant(
        {shape: convTransposeWInfo.shape, dataType: this.dataType_},
        convTransposeWData,
    );
    const convTransposeB = this.builder_.constant(
        {dataType: this.dataType_, shape: [1]},
        new this.ArrayType_([0.2734375]),
    );
    const convTranspose = this.builder_.convTranspose2d(add13, convTransposeW, {
      bias: convTransposeB,
      padding: this.layout === 'nhwc' ? [0, 0, 0, 0] : [0, 1, 0, 1],
      strides: [2, 2],
      outputSizes: [144, 256],
      filterLayout: this.layout === 'nhwc' ? 'ohwi' : 'iohw',
      inputLayout: this.layout,
    });

    // name: activation_10
    const sigmoid = this.builder_.sigmoid(convTranspose);
    if (this.layout === 'nhwc') {
      return sigmoid;
    } else {
      return this.builder_.reshape(sigmoid, this.outputShape_);
    }
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({segment_back: outputOperand});
  }

  async compute(inputBuffer) {
    this.context_.writeTensor(this.inputTensor_, inputBuffer);
    const inputs = {input: this.inputTensor_};
    const outputs = {segment_back: this.outputTensor_};
    this.context_.dispatch(this.graph_, inputs, outputs);
    const results = await this.context_.readTensor(this.outputTensor_);
    return new this.ArrayType_(results);
  }
}
