'use strict';

import {numpy} from './libs/numpy.js';
import {addAlert, checkBtn, handleBtnUI} from './ui.js';

export const isFloat16ArrayAvailable =
  typeof Float16Array !== 'undefined' && Float16Array.from;

export function weightsOrigin() {
  if (location.hostname.toLowerCase().indexOf('github.io') > -1) {
    return 'https://d3i5xkfad89fac.cloudfront.net';
  } else {
    return '..';
  }
}

export function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

// This function is used for reading buffer from a given url,
// which will be exported to node.js environment as well,
// so we use 'fs' module for examples ran in node.js and
// fetch() method for examples ran in browser.
export async function getBufferFromUrl(url) {
  let arrayBuffer;
  if (globalThis.fetch) {
    const response = await fetch(url);
    arrayBuffer = await response.arrayBuffer();
  } else {
    const fs = await import('fs');
    const uint8Array = await fs.promises.readFile(url);
    arrayBuffer = uint8Array.buffer;
  }
  return arrayBuffer;
}

// ref: http://stackoverflow.com/questions/32633585/how-do-you-convert-to-half-floats-in-javascript
export const toHalf = (function() {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);

  /* This method is faster than the OpenEXR implementation (very often
   * used, eg. in Ogre), with the additional benefit of rounding, inspired
   * by James Tursa?s half-precision code. */
  return function toHalf(val) {
    floatView[0] = val;
    const x = int32View[0];

    let bits = (x >> 16) & 0x8000; /* Get the sign */
    let m = (x >> 12) & 0x07ff; /* Keep one extra bit for rounding */
    const e = (x >> 23) & 0xff; /* Using int is faster here */

    /* If zero, or denormal, or exponent underflows too much for a denormal
     * half, return signed zero. */
    if (e < 103) {
      return bits;
    }

    /* If NaN, return NaN. If Inf or exponent overflow, return Inf. */
    if (e > 142) {
      bits |= 0x7c00;
      /* If exponent was 0xff and one mantissa bit was set, it means NaN,
       * not Inf, so make sure we set one mantissa bit too. */
      bits |= ((e == 255) ? 0 : 1) && (x & 0x007fffff);
      return bits;
    }

    /* If exponent underflows but not too much, return a denormal */
    if (e < 113) {
      m |= 0x0800;
      /* Extra rounding may overflow and set mantissa to 0 and exponent
       * to 1, which is OK. */
      bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
      return bits;
    }

    bits |= ((e - 112) << 10) | (m >> 1);
    /* Extra rounding. An overflow will set mantissa to 0 and increment
     * the exponent, which is OK. */
    bits += m & 1;
    return bits;
  };
})();

// Convert a float32 typed array to float16 data. Use Float16Array if it is
// available, otherwise use Uint16Array holding the float16 bit patterns.
export function toFloat16Array(float32Array) {
  return isFloat16ArrayAvailable ?
    Float16Array.from(float32Array, (v) => v) :
    Uint16Array.from(float32Array, (v) => toHalf(v));
}

// Build a float32 constant, or a float16 constant if `targetType` is
// 'float16'. `data` is a Float32Array (or array of numbers).
export function buildFloatConstant(builder, shape, data, targetType) {
  const float32Data =
    data instanceof Float32Array ? data : new Float32Array(data);
  if (targetType === 'float16') {
    return builder.constant(
        {dataType: 'float16', dimensions: shape, shape},
        toFloat16Array(float32Data));
  }
  return builder.constant(
      {dataType: 'float32', dimensions: shape, shape}, float32Data);
}

// Convert npy data in original data type to `targetType`, only support
// 'float32' to 'float16' conversion currently.
export async function buildConstantByNpy(builder, url, targetType = 'float32') {
  const dataTypeMap = new Map([
    ['f2', {type: 'float16',
      array: isFloat16ArrayAvailable ? Float16Array : Uint16Array}],
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
  const shape = npArray.shape;
  let type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  let typedArray = new TypedArrayConstructor(npArray.data.buffer);
  if (type === 'float32' && targetType === 'float16') {
    typedArray = toFloat16Array(typedArray);
    type = targetType;
  } else if (type !== targetType) {
    throw new Error(`Conversion from ${npArray.dataType} ` +
        `to ${targetType} is not supported.`);
  }
  return builder.constant(
      {dataType: type, dimensions: shape, shape}, typedArray);
}

// Convert video frame to a canvas element
export function getVideoFrame(videoElement) {
  const canvasElement = document.createElement('canvas');
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  const canvasContext = canvasElement.getContext('2d');
  canvasContext.drawImage(videoElement, 0, 0, canvasElement.width,
      canvasElement.height);
  return canvasElement;
}

// Get media stream from camera
export async function getMediaStream(resolution = {}) {
  const height = undefined || resolution.height;
  const width = undefined || resolution.width;
  // Support 'user' facing mode at present
  const constraints =
      {audio: false, video: {facingMode: 'user', height, width}};
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  return stream;
}

// Stop camera stream and cancel animation frame
export function stopCameraStream(id, stream) {
  cancelAnimationFrame(id);
  if (stream) {
    stream.getTracks().forEach((track) => {
      if (track.readyState === 'live' && track.kind === 'video') {
        track.stop();
      }
    });
  }
}

/**
 * This method is used to covert input element to tensor data.
 * @param {Object} inputElement, an object of HTML [<img> | <video>] element.
 * @param {!Object<string, *>} inputOptions, an object of options to process
 * input element.
 * inputOptions = {
 *     inputLayout {String}, // input layout of tensor.
 *     inputShape: {!Array<number>}, // shape of input tensor.
 *     mean: {Array<number>}, // optional, mean values for processing the input
 *       element. If not specified, it will be set to [0, 0, 0, 0].
 *     std: {Array<number>}, // optional, std values for processing the input
 *       element. If not specified, it will be set to [1, 1, 1, 1].
 *     norm: {Boolean}, // optional, normlization flag. If not specified,
 *       it will be set to false.
 *     scaledFlag: {boolean}, // optional, scaling flag. If specified,
 *       scale the width and height of the input element.
 *     drawOptions: { // optional, drawOptions is used for
 *         CanvasRenderingContext2D.drawImage() method.
 *       sx: {number}, // the x-axis coordinate of the top left corner of
 *         sub-retangle of the source image.
 *       sy: {number}, // the y-axis coordinate of the top left corner of
 *         sub-retangle of the source image.
 *       sWidth: {number}, // the width of the sub-retangle of the
 *         source image.
 *       sHeight: {number}, // the height of the sub-retangle of the
 *         source image.
 *       dWidth: {number}, // the width to draw the image in the detination
 *         canvas.
 *       dHeight: {number}, // the height to draw the image in the detination
 *         canvas.
 *     },
 * };
 * @return {Object} tensor, an object of input tensor.
 */
export function getInputTensor(inputElement, inputOptions) {
  const inputShape = inputOptions.inputShape;
  const tensor = new Float32Array(
      inputShape.slice(1).reduce((a, b) => a * b));

  inputElement.width = inputElement.videoWidth ||
      inputElement.naturalWidth;
  inputElement.height = inputElement.videoHeight ||
      inputElement.naturalHeight;

  let [channels, height, width] = inputShape.slice(1);
  const mean = inputOptions.mean || [0, 0, 0, 0];
  const std = inputOptions.std || [1, 1, 1, 1];
  const normlizationFlag = inputOptions.norm || false;
  const channelScheme = inputOptions.channelScheme || 'RGB';
  const scaledFlag = inputOptions.scaledFlag || false;
  const inputLayout = inputOptions.inputLayout;
  const imageChannels = 4; // RGBA
  const drawOptions = inputOptions.drawOptions;
  if (inputLayout === 'nhwc') {
    [height, width, channels] = inputShape.slice(1);
  }
  const canvasElement = document.createElement('canvas');
  canvasElement.width = width;
  canvasElement.height = height;
  const canvasContext = canvasElement.getContext('2d');

  if (drawOptions) {
    canvasContext.drawImage(inputElement, drawOptions.sx, drawOptions.sy,
        drawOptions.sWidth, drawOptions.sHeight, 0, 0, drawOptions.dWidth,
        drawOptions.dHeight);
  } else {
    if (scaledFlag) {
      const resizeRatio = Math.max(Math.max(
          inputElement.width / width, inputElement.height / height), 1);
      const scaledWidth = Math.floor(inputElement.width / resizeRatio);
      const scaledHeight = Math.floor(inputElement.height / resizeRatio);
      canvasContext.drawImage(inputElement, 0, 0, scaledWidth, scaledHeight);
    } else {
      canvasContext.drawImage(inputElement, 0, 0, width, height);
    }
  }

  let pixels = canvasContext.getImageData(0, 0, width, height).data;

  if (normlizationFlag) {
    pixels = new Float32Array(pixels).map((p) => p / 255);
  }

  for (let c = 0; c < channels; ++c) {
    for (let h = 0; h < height; ++h) {
      for (let w = 0; w < width; ++w) {
        let value;
        if (channelScheme === 'BGR') {
          value = pixels[h * width * imageChannels + w * imageChannels +
              (channels - c - 1)];
        } else {
          value = pixels[h * width * imageChannels + w * imageChannels + c];
        }
        if (inputLayout === 'nchw') {
          tensor[c * width * height + h * width + w] =
              (value - mean[c]) / std[c];
        } else {
          tensor[h * width * channels + w * channels + c] =
              (value - mean[c]) / std[c];
        }
      }
    }
  }
  return tensor;
}

// Get median value from an array of Number
export function getMedianValue(array) {
  array = array.sort((a, b) => a - b);
  return array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
      (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
}

// Get url params
export function getUrlParams() {
  const params = new URLSearchParams(location.search);
  const lowerCaseParams = {};
  params.forEach((value, key) => {
    lowerCaseParams[key.toLowerCase()] = value;
  });

  // Get 'numRuns' param to run inference multiple times
  let numRuns = lowerCaseParams['numruns'];
  numRuns = numRuns ? parseInt(numRuns) : 1;
  if (numRuns < 1) {
    addAlert(`Ignore the url param: 'numRuns', its value must be >= 1.`);
    numRuns = 1;
  }

  // Get 'powerPreference' param to set WebNN's 'MLPowerPreference' option
  let powerPreference = lowerCaseParams['powerpreference'];
  const powerPreferences = ['default', 'high-performance', 'low-power'];

  if (powerPreference && !powerPreferences.includes(powerPreference)) {
    addAlert(`Ignore the url param: 'powerPreference', its value must be ` +
        `one of {'default', 'high-performance', 'low-power'}.`);
    powerPreference = null;
  }

  return [numRuns, powerPreference];
}

export async function isWebNN() {
  if (typeof MLGraphBuilder !== 'undefined') {
    // Polyfill MLTensorUsage to make it compatible with old version of Chrome.
    if (typeof MLTensorUsage == 'undefined') {
      window.MLTensorUsage = {WEBGPU_INTEROP: 1, READ: 2, WRITE: 4};
    }
    return true;
  } else {
    return false;
  }
}

export function webNNNotSupportMessage() {
  return 'Your browser does not support WebNN.';
}

export function webNNNotSupportMessageHTML() {
  return 'Your browser does not support WebNN. Please refer to <a href="https://github.com/webmachinelearning/webnn-samples/#webnn-installation-guides">WebNN Installation Guides</a> for more details.';
}

// Derive from
// https://github.com/webmachinelearning/webnn-baseline/blob/main/src/lib/compute-padding.js
/**
 * Compute the beginning and ending pad given input, filter and stride sizes.
 * @param {String} autoPad
 * @param {Number} inputSize
 * @param {Number} effectiveFilterSize
 * @param {Number} stride
 * @param {Number} outputPadding
 * @return {Array} [paddingBegin, paddingEnd]
 */
function computePadding1DForAutoPad(
    autoPad, inputSize, effectiveFilterSize, stride, outputPadding) {
  let totalPadding;
  if (outputPadding === undefined) {
    // for conv2d
    const outSize = Math.ceil(inputSize / stride);
    const neededInput = (outSize - 1) * stride + effectiveFilterSize;
    totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
  } else {
    // for convTranspose2d
    // totalPadding = beginning padding + ending padding
    // SAME_UPPER or SAME_LOWER mean pad the input so that
    //   output size = input size * strides
    // output size = (input size - 1) * stride + effectiveFilterSize
    //     - beginning padding - ending padding + output padding
    totalPadding = (inputSize - 1) * stride + effectiveFilterSize +
        outputPadding - inputSize * stride;
  }
  let paddingBegin;
  let paddingEnd;
  switch (autoPad) {
    case 'same-upper':
      paddingBegin = Math.floor(totalPadding / 2);
      paddingEnd = Math.floor((totalPadding + 1) / 2);
      break;
    case 'same-lower':
      paddingBegin = Math.floor((totalPadding + 1) / 2);
      paddingEnd = Math.floor(totalPadding / 2);
      break;
    default:
      throw new Error('The autoPad is invalid.');
  }
  return [paddingBegin, paddingEnd];
}

// Compute explicit padding given input sizes, filter sizes, strides, dilations
// and auto pad mode 'same-upper' or 'same-lower'.
export function computePadding2DForAutoPad(
    inputSizes, filterSizes, strides, dilations, autoPad) {
  const [inputHeight, inputWidth] = inputSizes;
  const [filterHeight, filterWidth] = filterSizes;
  const [strideHeight, strideWidth] = strides ? strides : [1, 1];
  const [dilationHeight, dilationWidth] = dilations ? dilations: [1, 1];
  const effectiveFilterHeight = (filterHeight - 1) * dilationHeight + 1;
  const effectiveFilterWidth = (filterWidth - 1) * dilationWidth + 1;
  const [beginningPaddingHeight, endingPaddingHeight] =
      computePadding1DForAutoPad(
          autoPad, inputHeight, effectiveFilterHeight, strideHeight);
  const [beginningPaddingWidth, endingPaddingWidth] =
      computePadding1DForAutoPad(
          autoPad, inputWidth, effectiveFilterWidth, strideWidth);
  return [beginningPaddingHeight, endingPaddingHeight,
    beginningPaddingWidth, endingPaddingWidth];
}

// This function derives from Transformer.js `permute_data()` function:
// https://github.com/xenova/transformers.js/blob/main/src/utils/maths.js#L98
// which is in Apache License 2.0
// https://github.com/xenova/transformers.js/blob/main/LICENSE
/**
 * Helper method to permute a `AnyTypedArray` directly
 * @template {AnyTypedArray} T
 * @param {T} array
 * @param {number[]} dims
 * @param {number[]} axes
 * @return {[T, number[]]} The permuted array and the new shape.
 */
export function permuteData(array, dims, axes) {
  // Calculate the new shape of the permuted array
  // and the stride of the original array
  const shape = new Array(axes.length);
  const stride = new Array(axes.length);

  for (let i = axes.length - 1, s = 1; i >= 0; --i) {
    stride[i] = s;
    shape[i] = dims[axes[i]];
    s *= shape[i];
  }

  // Precompute inverse mapping of stride
  const invStride = axes.map((_, i) => stride[axes.indexOf(i)]);

  // Create the permuted array with the new shape
  // @ts-ignore
  const permutedData = new array.constructor(array.length);

  // Permute the original array to the new array
  for (let i = 0; i < array.length; ++i) {
    let newIndex = 0;
    for (let j = dims.length - 1, k = i; j >= 0; --j) {
      newIndex += (k % dims[j]) * invStride[j];
      k = Math.floor(k / dims[j]);
    }
    permutedData[newIndex] = array[i];
  }

  return [permutedData, shape];
}

const deviceLimitsCache = new Map();

async function queryDeviceLimits(deviceType) {
  const context = await navigator.ml.createContext({deviceType});
  try {
    return context.opSupportLimits();
  } finally {
    context.destroy?.();
  }
}

/**
 * Get opSupportLimits() for a device. Creates at most one throwaway context
 * per device for the lifetime of the page.
 * @param {String} deviceType 'cpu', 'gpu' or 'npu'.
 * @return {Object|null} the limits, or null if they can't be queried. The
 *     underlying error resurfaces when the sample creates its own context.
 */
async function getDeviceLimits(deviceType) {
  if (!deviceLimitsCache.has(deviceType)) {
    deviceLimitsCache.set(deviceType, queryDeviceLimits(deviceType));
  }
  try {
    return await deviceLimitsCache.get(deviceType);
  } catch (error) {
    deviceLimitsCache.delete(deviceType);
    console.warn(`Unable to query WebNN ${deviceType} limits: ${error}`);
    return null;
  }
}

export async function getDefaultLayout(deviceType) {
  const limits = await getDeviceLimits(deviceType);
  return limits?.preferredInputLayout ?? 'nchw';
}

/**
 * Get the floating point data types ('float32' and/or 'float16') the device
 * can likely run the sample models with, based on its opSupportLimits().
 * This is a heuristic: it checks graph inputs/outputs, constants, cast and
 * conv2d, not every op a model uses, so building a graph can still fail.
 *
 * By default float16 models are assumed to take float32 graph inputs/outputs
 * and cast to/from float16 inside the graph. Pass `nativeFloat16IO` for
 * models whose graph inputs/outputs are float16 (e.g. selfie_segmentation).
 *
 * @param {String} deviceType 'cpu', 'gpu' or 'npu'.
 * @param {Object} options
 *     nativeFloat16IO: Boolean, see above.
 *     preferredDataType: String, listed first if supported. Defaults to
 *         'float16' for NPU (which generally runs float16 natively) and
 *         'float32' otherwise.
 * @return {Array} supported data types, preferred first. Both are returned
 *     if the device's limits can't be queried.
 */
export async function getSupportedDataTypes(deviceType, {
  nativeFloat16IO = false,
  preferredDataType = deviceType === 'npu' ? 'float16' : 'float32',
} = {}) {
  const limits = await getDeviceLimits(deviceType);
  let supportedDataTypes = ['float32', 'float16'];
  if (limits) {
    // Assume support if an implementation doesn't report a limit.
    const supports = (limit, dataType) =>
      limit?.dataTypes?.includes(dataType) ?? true;
    const float32IO = supports(limits.input, 'float32') &&
      supports(limits.output, 'float32');
    const float32 = float32IO && supports(limits.constant, 'float32') &&
      supports(limits.conv2d?.input, 'float32');
    const float16IO = nativeFloat16IO ?
      supports(limits.input, 'float16') &&
        supports(limits.output, 'float16') :
      float32IO && supports(limits.cast?.input, 'float32') &&
        supports(limits.cast?.input, 'float16');
    const float16 = float16IO && supports(limits.constant, 'float16') &&
      supports(limits.conv2d?.input, 'float16');
    supportedDataTypes = supportedDataTypes.filter(
        (type) => type === 'float32' ? float32 : float16);
  }
  return supportedDataTypes.sort(
      (a, b) => (b === preferredDataType) - (a === preferredDataType));
}

/**
 * On a device change, enable only the Float32/Float16 buttons
 * (#float32Label/#float16Label) the device supports and select its
 * preferred data type, without triggering the buttons' change handlers.
 * @param {String} deviceType 'cpu', 'gpu' or 'npu'.
 * @param {Object} options passed to getSupportedDataTypes().
 * @return {String|null} the selected data type, or null (after showing an
 *     alert) if the device supports neither.
 */
export async function selectDataTypeForDevice(deviceType, options = {}) {
  const supportedDataTypes =
    await getSupportedDataTypes(deviceType, options);
  // eslint-disable-next-line no-unused-vars
  for (const type of ['float32', 'float16']) {
    handleBtnUI(`#${type}Label`, !supportedDataTypes.includes(type));
  }
  if (supportedDataTypes.length === 0) {
    addAlert(`WebNN (${deviceType.toUpperCase()}) supports neither ` +
      'float32 nor float16 for this sample.');
    return null;
  }
  checkBtn(supportedDataTypes[0]);
  return supportedDataTypes[0];
}

/**
 * Display available models based on device type and data type.
 * @param {Object} modelList list of available models.
 * @param {Array} modelIds list of model ids.
 * @param {String} layout 'nchw' or 'nhwc'.
 * @param {String} dataType 'float32', 'float16', or ''.
 */
export function displayAvailableModels(modelList, modelIds, layout, dataType) {
  let models = [];
  if (dataType == '') {
    models = models.concat(modelList[layout]['float32'] ?? []);
    models = models.concat(modelList[layout]['float16'] ?? []);
    models = models.concat(modelList[layout]['uint8'] ?? []);
  } else {
    models = models.concat(modelList[layout][dataType] ?? []);
  }
  // Remove duplicate ids.
  models = [...new Set(models)];
  // Display available models.
  // eslint-disable-next-line no-unused-vars
  for (const modelId of modelIds) {
    if (models.includes(modelId)) {
      $(`#${modelId}`).parent().show();
    } else {
      $(`#${modelId}`).parent().hide();
    }
  }
}
