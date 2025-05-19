'use strict';

import {buildConstantByNpy, computePadding2DForAutoPad, weightsOrigin} from '../common/utils.js';

/* eslint max-len: ["error", {"code": 120}] */

// MobileNet V2 model with 'nhwc' input layout
export class MobileNetV2Uint8Nhwc {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.weightsUrl_ = weightsOrigin() +
      '/test-data/models/mobilenetv2_nhwc/weights_uint8/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: 'nhwc',
      labelUrl: './labels/labels1001.txt',
      inputShape: [1, 224, 224, 3],
    };
    this.outputShape_ = [1, 1001];
  }

  dequantizeLinear_(input, quantizateParams, dataType) {
    const scale = this.builder_.constant( {dataType: 'float32', shape: quantizateParams.shape}, 
        new Float32Array(quantizateParams.scale));
    let zeroPoint;
    if (dataType === 'uint8') {
      zeroPoint = this.builder_.constant( {dataType: 'uint8', shape: quantizateParams.shape},
        new Uint8Array(quantizateParams.zero_point));
    } else if (dataType === 'int32') {
      zeroPoint = this.builder_.constant( {dataType: 'int32', shape: quantizateParams.shape},
        new Int32Array(quantizateParams.zero_point));
    } else {
      throw new Error(`Data type ${dataType} is not supported.`);
    }
    return this.builder_.dequantizeLinear(input, scale, zeroPoint);
  }

  quantizeLinear_(input, quantizateParams) {
    const scale = this.builder_.constant( {dataType: 'float32', shape: quantizateParams.shape}, 
        new Float32Array(quantizateParams.scale));
    const zeroPoint = this.builder_.constant( {dataType: 'uint8', shape: quantizateParams.shape},
      new Uint8Array(quantizateParams.zero_point));
    return this.builder_.quantizeLinear(input, scale, zeroPoint);
  }

  async buildConv_(input, weightsSubName, weightsquantize, biasquantize, outputQuantize, relu6, options) {
    const weightsName = this.weightsUrl_ + 'weights_' + weightsSubName + '.npy';
    const weights = await buildConstantByNpy(this.builder_, weightsName, 'uint8');
    const dequantizeWeights = this.dequantizeLinear_(weights, weightsquantize, 'uint8');
    const biasName = this.weightsUrl_ + 'bias_' + weightsSubName + '.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName, 'int32');
    const dequantizeBias = this.dequantizeLinear_(bias, biasquantize, 'int32');
    options.inputLayout = 'nhwc';
    options.bias = dequantizeBias;
    // WebNN spec drops autoPad support, compute the explicit padding instead.
    if (options.autoPad == 'same-upper') {
      const isShapeMethod = typeof weights.shape === 'function';
      const inputShape = isShapeMethod ? (await input).shape() : (await input).shape;
      const weightsShape = isShapeMethod ? weights.shape() : weights.shape;
      options.padding =
        computePadding2DForAutoPad(
            /* nwhc */[inputShape[1], inputShape[2]],
            /* ohwi or ihwo */[weightsShape[1], weightsShape[2]],
            options.strides, options.dilations, options.autoPad);
    }
    const conv2d = this.builder_.conv2d(input, dequantizeWeights, options);
    if (relu6) {
      // `relu6` in TFLite equals to `clamp` in WebNN API
      const clamp = this.builder_.clamp(conv2d, {minValue: 0, maxValue: 6});
      const quantizeClamp = this.quantizeLinear_(clamp, outputQuantize);
      return this.dequantizeLinear_(quantizeClamp, outputQuantize, 'uint8');
    }
    const quantizeConv2d = this.quantizeLinear_(conv2d, outputQuantize);
    return this.dequantizeLinear_(quantizeConv2d, outputQuantize, 'uint8');
  }

  async buildLinearBottleneck_(input, weightsNameArray, weightsquantizeArray, biasquantizeArray,
      outputquantizeArray, addQuantize, dwiseOptions, shortcut = true) {
    const autoPad = 'same-upper';

    const convOptions = {autoPad, filterLayout: 'ohwi'};
    const conv1x1Relu6 = await this.buildConv_(
        input, weightsNameArray[0], weightsquantizeArray[0], biasquantizeArray[0],
        outputquantizeArray[0], true, convOptions);

    dwiseOptions.autoPad = autoPad;
    dwiseOptions.filterLayout = 'ihwo';
    const dwise3x3Relu6 = await this.buildConv_(
        conv1x1Relu6, weightsNameArray[1], weightsquantizeArray[1], biasquantizeArray[1],
        outputquantizeArray[1], true, dwiseOptions);

    const conv1x1Linear = await this.buildConv_(
        dwise3x3Relu6, weightsNameArray[2], weightsquantizeArray[2], biasquantizeArray[2],
        outputquantizeArray[2], false, convOptions);

    if (shortcut) {
      const add = this.builder_.add(input, conv1x1Linear);
      const quantizeAdd = this.quantizeLinear_(add, addQuantize);
      return this.dequantizeLinear_(quantizeAdd, addQuantize, 'uint8');
    }
    return conv1x1Linear;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const strides = [2, 2];
    const autoPad = 'same-upper';
    const filterLayout = 'ohwi';
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
    const conv0 = await this.buildConv_(
        input, '55', 
        {scale: [0.034375786781311035], zero_point: [159], shape: []},
        {scale: [0.0002706754894461483], zero_point: [0], shape: []},
        {scale: [0.02352987229824066], zero_point: [0], shape: []},
        true, {strides, autoPad, filterLayout});
    const conv1 = await this.buildConv_(
        conv0, '57', 
        {scale: [0.5174643397331238], zero_point: [115], shape: []},
        {scale: [0.012175869196653366], zero_point: [0], shape: []}, 
        {scale: [0.02352987229824066], zero_point: [0], shape: []},
        true, {autoPad, groups: 32, filterLayout: 'ihwo'});
    const conv2 = await this.buildConv_(
        conv1, '59', 
        {scale: [0.06309328228235245], zero_point: [90], shape: []},
        {scale: [0.0014845768455415964], zero_point: [0], shape: []}, 
        {scale: [0.3935633599758148], zero_point: [129], shape: []},
        false, {autoPad, filterLayout});
    const bottleneck0 = await this.buildLinearBottleneck_(
        conv2, ['61', '63', '65'], 
        [
          {scale: [0.008153429254889488], zero_point: [85], shape: []},
          {scale: [0.01082384679466486], zero_point: [118], shape: []},
          {scale: [0.03367125615477562], zero_point: [152], shape: []}
        ],
        [
          {scale: [0.003208891022950411], zero_point: [0], shape: []},
          {scale: [0.00025468372041359544], zero_point: [0], shape: []},
          {scale: [0.0007922803633846343], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066 ], zero_point: [0], shape: []},
          {scale: [0.32650214433670044], zero_point: [117], shape: []}
        ],
        {}, {strides, groups: 96}, false);
    const bottleneck1 = await this.buildLinearBottleneck_(
        bottleneck0, ['67', '69', '71'], 
        [
          {scale: [0.003573313821107149], zero_point: [102], shape: []},
          {scale: [0.14301884174346924], zero_point: [166], shape: []},
          {scale: [0.0644076019525528], zero_point: [122], shape: []}
        ],
        [
          {scale: [0.0011666945647448301], zero_point: [0], shape: []},
          {scale: [0.003365214914083481], zero_point: [0], shape: []},
          {scale: [0.0015155026922002435], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.6063596606254578], zero_point: [111], shape: []}
        ],
        {scale: [0.609800398349762], zero_point: [117], shape: []},
        {groups: 144});
    const bottleneck2 = await this.buildLinearBottleneck_(
        bottleneck1, ['74', '76', '78'],
        [
          {scale: [0.002144400728866458], zero_point: [129], shape: []},
          {scale: [0.026209063827991486], zero_point: [141], shape: []},
          {scale: [0.030137652531266212], zero_point: [160], shape: []}
        ],
        [
          {scale: [0.0013076564064249396], zero_point: [0], shape: []},
          {scale: [0.0006166959065012634], zero_point: [0], shape: []},
          {scale: [0.0007091350853443146], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.28959763050079346], zero_point: [132], shape: []}
        ],
        {}, {strides, groups: 144}, false);
    const bottleneck3 = await this.buildLinearBottleneck_(
        bottleneck2, ['80', '82', '84'], 
        [
          {scale: [0.0018756906501948833], zero_point: [129], shape: []},
          {scale: [0.04708000645041466], zero_point: [107], shape: []},
          {scale: [0.021888649091124535], zero_point: [144], shape: []}
        ],
        [
          {scale: [0.0005431955796666443], zero_point: [0], shape: []},
          {scale: [0.0011077865492552519], zero_point: [0], shape: []},
          {scale: [0.0005150370998308063], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.2785290479660034], zero_point: [144], shape: []}
        ],
        {scale: [0.3821489214897156], zero_point: [134], shape: []},
        {groups: 192});
    const bottleneck4 = await this.buildLinearBottleneck_(
        bottleneck3, ['87', '89', '91'], 
        [
          {scale: [0.001845767954364419 ], zero_point: [144], shape: []},
          {scale: [0.056680627167224884 ], zero_point: [138], shape: []},
          {scale: [0.027344657108187675], zero_point: [141], shape: []}
        ],
        [
          {scale: [0.0007053582230582833], zero_point: [0], shape: []},
          {scale: [0.0013336879201233387], zero_point: [0], shape: []},
          {scale: [0.0006434162496589124], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.31692883372306824], zero_point: [130], shape: []}
        ],
        {scale: [0.4749276041984558], zero_point: [122], shape: []},
        {groups: 192});
    const bottleneck5 = await this.buildLinearBottleneck_(
        bottleneck4, ['94', '96', '98'], 
        [
          {scale: [0.0021686102263629436], zero_point: [147], shape: []},
          {scale: [0.01276324037462473], zero_point: [141], shape: []},
          {scale: [0.01878936029970646], zero_point: [145], shape: []}
        ],
        [
          {scale: [0.0010299327550455928], zero_point: [0], shape: []},
          {scale: [0.00030031739152036607], zero_point: [0], shape: []},
          {scale: [0.0004421112244017422], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.2426074892282486], zero_point: [126], shape: []}
        ],
        {}, {strides, groups: 192}, false);
    const bottleneck6 = await this.buildLinearBottleneck_(
        bottleneck5, ['100', '102', '104'], 
        [
          {scale: [0.0018693081801757216], zero_point: [124], shape: []},
          {scale: [0.057145655155181885], zero_point: [131], shape: []},
          {scale: [0.024178611114621162], zero_point: [173], shape: []}
        ],
        [
          {scale: [0.0004535081679932773], zero_point: [0], shape: []},
          {scale: [0.0013446299126371741], zero_point: [0], shape: []},
          {scale: [0.0005689196405000985], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.24107083678245544], zero_point: [99], shape: []}
        ], 
        {scale: [0.272112101316452], zero_point: [122], shape: []},
        {groups: 384});
    const bottleneck7 = await this.buildLinearBottleneck_(
        bottleneck6, ['107', '109', '111'],
        [
          {scale: [0.0013072594301775098], zero_point: [139], shape: []},
          {scale: [0.03875831514596939], zero_point: [143], shape: []},
          {scale: [0.021180255338549614], zero_point: [145], shape: []}
        ],
        [
          {scale: [0.0003557211020961404], zero_point: [0], shape: []},
          {scale: [0.0009119781898334622], zero_point: [0], shape: []},
          {scale: [0.0004983686958439648], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.21128740906715393], zero_point: [133], shape: []}
        ], 
        {scale: [0.30632293224334717], zero_point: [119], shape: []},
        {groups: 384});
    const bottleneck8 = await this.buildLinearBottleneck_(
        bottleneck7, ['114', '116', '118'],
        [
          {scale: [0.0011219490552321076], zero_point: [138], shape: []},
          {scale: [0.03533448651432991], zero_point: [107], shape: []},
          {scale: [0.025988703593611717 ], zero_point: [151], shape: []}
        ],
        [
          {scale: [0.0003436787228565663], zero_point: [0], shape: []},
          {scale: [0.0008314159349538386], zero_point: [0], shape: []},
          {scale: [0.0006115108844824135], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.2665506601333618], zero_point: [126], shape: []}
        ], 
        {scale: [0.3084481954574585], zero_point: [130], shape: []},
        {groups: 384});
    const bottleneck9 = await this.buildLinearBottleneck_(
        bottleneck8, ['121', '123', '125'], 
        [
          {scale: [0.0015335703501477838], zero_point: [156], shape: []},
          {scale: [0.02276834286749363], zero_point: [131], shape: []},
          {scale: [0.012576368637382984], zero_point: [100], shape: []}
        ],
        [
          {scale: [0.00047302700113505125], zero_point: [0], shape: []},
          {scale: [0.0005357362097129226], zero_point: [0], shape: []},
          {scale: [0.0002959203557111323], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.2215471863746643], zero_point: [126], shape: []}
        ], 
        {}, {groups: 384}, false);
    const bottleneck10 = await this.buildLinearBottleneck_(
        bottleneck9, ['127', '129', '131'],
        [
          {scale: [0.0014903603587299585], zero_point: [110], shape: []},
          {scale: [0.04933452978730202], zero_point: [131], shape: []},
          {scale: [0.012083801440894604], zero_point: [152], shape: []}
        ],
        [
          {scale: [0.0003301851393189281], zero_point: [0], shape: []},
          {scale: [0.0011608351487666368], zero_point: [0], shape: []},
          {scale: [0.0002843302791006863], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.1737687587738037], zero_point: [126], shape: []}
        ],
        {scale: [0.23995822668075562], zero_point: [128], shape: []},
        {groups: 576});
    const bottleneck11 = await this.buildLinearBottleneck_(
        bottleneck10, ['134', '136', '138'], 
        [
          {scale: [0.0030131470412015915], zero_point: [142], shape: []},
          {scale: [0.09067106992006302], zero_point: [107], shape: []},
          {scale: [0.01852469891309738], zero_point: [123], shape: []}
        ],
        [
          {scale: [0.000723029428627342], zero_point: [0], shape: []},
          {scale: [0.002133478643372655], zero_point: [0], shape: []},
          {scale: [0.00043588379048742354], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.2431039810180664], zero_point: [129], shape: []}
        ],
        {scale: [0.3369702398777008], zero_point: [125], shape: []},
        {groups: 576});
    const bottleneck12 = await this.buildLinearBottleneck_(
        bottleneck11, ['141', '143', '145'],
        [
          {scale: [0.0016000346513465047 ], zero_point: [123], shape: []},
          {scale: [0.06790248304605484], zero_point: [69], shape: []},
          {scale: [0.01406034268438816], zero_point: [149], shape: []}
        ],
        [
          {scale: [0.0005391640588641167], zero_point: [0], shape: []},
          {scale: [0.001597736612893641], zero_point: [0], shape: []},
          {scale: [0.0003308380546513945], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.18441903591156006], zero_point: [137], shape: []}
        ],
        {}, {strides, groups: 576}, false);
    const bottleneck13 = await this.buildLinearBottleneck_(
        bottleneck12, ['147', '149', '151'], 
        [
          {scale: [0.002654522191733122], zero_point: [118], shape: []},
          {scale: [0.0386493057012558], zero_point: [148], shape: []},
          {scale: [0.012022278271615505], zero_point: [152], shape: []}
        ],
        [
          {scale: [0.0004895444144494832], zero_point: [0], shape: []},
          {scale: [0.0009094132692553103], zero_point: [0], shape: []},
          {scale: [0.0002828826545737684], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.15274383127689362 ], zero_point: [117], shape: []}
        ],
        {scale: [0.18867115676403046], zero_point: [125], shape: []},
        {groups: 960});
    const bottleneck14 = await this.buildLinearBottleneck_(
        bottleneck13, ['154', '156', '158'], 
        [
          {scale: [0.002872081007808447], zero_point: [171], shape: []},
          {scale: [0.042505424469709396], zero_point: [174], shape: []},
          {scale: [0.07219446450471878], zero_point: [89], shape: []}
        ],
        [
          {scale: [0.0005418788641691208], zero_point: [0], shape: []},
          {scale: [0.0010001471964642406], zero_point: [0], shape: []},
          {scale: [0.0016987264389172196], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.46753087639808655], zero_point: [168], shape: []}
        ],
        {scale: [0.5439051389694214], zero_point: [175], shape: []},
        {groups: 960});
    const bottleneck15 = await this.buildLinearBottleneck_(
        bottleneck14, ['161', '163', '165'],
        [
          {scale: [0.0015946601051837206], zero_point: [115], shape: []},
          {scale: [0.05400737002491951], zero_point: [154], shape: []},
          {scale: [0.025818506255745888], zero_point: [67], shape: []}
        ],
        [
          {scale: [0.0008673437987454236], zero_point: [0], shape: []},
          {scale: [0.001270786509849131], zero_point: [0], shape: []},
          {scale: [0.0006075061392039061], zero_point: [0], shape: []}
        ],
        [
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          {scale: [0.25913476943969727], zero_point: [176], shape: []}
        ],
        {}, {groups: 960}, false);
   
    const conv3 = await this.buildConv_(
          bottleneck15, '167', 
          {scale: [0.003932334017008543], zero_point: [116], shape: []},
          {scale: [0.001019004499539733], zero_point: [0], shape: []}, 
          {scale: [0.02352987229824066], zero_point: [0], shape: []},
          true, {autoPad, filterLayout});
  
    const poolQuantize = {scale: [0.02352987229824066], zero_point: [0], shape: []};
    const averagePool2d = this.builder_.averagePool2d(
      conv3, {windowDimensions: [7, 7], layout: 'nhwc'});
   
    const quantize2 = this.quantizeLinear_(averagePool2d, poolQuantize);
    const dequantize = this.dequantizeLinear_(quantize2, poolQuantize, 'uint8');
    const conv4 = await this.buildConv_(
      dequantize, '170', 
          {scale: [0.002771724946796894], zero_point: [105], shape: []},
          {scale: [0.00006521832983708009], zero_point: [0], shape: []}, 
          {scale: [0.06046031787991524], zero_point: [60], shape: []},
          false, {autoPad, filterLayout});
    const reshape = this.builder_.reshape(conv4, [1, 1001]);
    const softmax = this.builder_.softmax(reshape);
    
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
