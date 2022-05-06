'use strict';

import {numpy} from './libs/numpy.js';
import {addAlert} from './ui.js';

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

export async function buildConstantByNpy(builder, url) {
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

/**
 * This method is used to covert input element to tensor data.
 * @param {Object} inputElement, an object of HTML [<img> | <video>] element.
 * @param {!Object<string, *>} inputOptions, an object of options to process
 * input element.
 * inputOptions = {
 *     inputLayout {String}, // input layout of tensor.
 *     inputDimensions: {!Array<number>}, // dimensions of input tensor.
 *     mean: {Array<number>}, // optional, mean values for processing the input
 *       element. If not specified, it will be set to [0, 0, 0, 0].
 *     std: {Array<number>}, // optional, std values for processing the input
 *       element. If not specified, it will be set to [1, 1, 1, 1].
 *     norm: {Boolean}, // optional, normlization flag. If not specified,
 *       it will be set to false.
 *     scaledFlag: {boolean}, // optional, scaling flag. If specified,
 *       scale the width and height of the input element.
 * };
 * @return {Object} tensor, an object of input tensor.
 */
export function getInputTensor(inputElement, inputOptions) {
  const inputDimensions = inputOptions.inputDimensions;
  const tensor = new Float32Array(
      inputDimensions.slice(1).reduce((a, b) => a * b));

  inputElement.width = inputElement.videoWidth ||
      inputElement.naturalWidth;
  inputElement.height = inputElement.videoHeight ||
      inputElement.naturalHeight;

  let [channels, height, width] = inputDimensions.slice(1);
  const mean = inputOptions.mean || [0, 0, 0, 0];
  const std = inputOptions.std || [1, 1, 1, 1];
  const normlizationFlag = inputOptions.norm || false;
  const scaledFlag = inputOptions.scaledFlag || false;
  const inputLayout = inputOptions.inputLayout;
  const imageChannels = 4; // RGBA

  if (inputLayout === 'nhwc') {
    [height, width, channels] = inputDimensions.slice(1);
  }
  const canvasElement = document.createElement('canvas');
  canvasElement.width = width;
  canvasElement.height = height;
  const canvasContext = canvasElement.getContext('2d');

  if (scaledFlag) {
    const resizeRatio = Math.max(
        Math.max(inputElement.width / width, inputElement.height / height), 1);
    const scaledWidth = Math.floor(inputElement.width / resizeRatio);
    const scaledHeight = Math.floor(inputElement.height / resizeRatio);
    canvasContext.drawImage(inputElement, 0, 0, scaledWidth, scaledHeight);
  } else {
    canvasContext.drawImage(inputElement, 0, 0, width, height);
  }

  let pixels = canvasContext.getImageData(0, 0, width, height).data;

  if (normlizationFlag) {
    pixels = new Float32Array(pixels).map((p) => p / 255);
  }

  for (let c = 0; c < channels; ++c) {
    for (let h = 0; h < height; ++h) {
      for (let w = 0; w < width; ++w) {
        const value =
            pixels[h * width * imageChannels + w * imageChannels + c];
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

// Set tf.js backend based WebNN's 'MLDevicePreference' option
export async function setPolyfillBackend(device) {
  // Simulate WebNN's device selection using various tf.js backends.
  // MLDevicePreference: ['default', 'gpu', 'cpu']
  // 'default' or 'gpu': tfjs-backend-webgl, 'cpu': tfjs-backend-wasm
  if (!device) device = 'gpu';
  // Use 'webgl' by default for better performance.
  // Note: 'wasm' backend may run failed on some samples since
  // some ops aren't supported on 'wasm' backend at present
  const backend = device === 'cpu' ? 'wasm' : 'webgpu';
  const tf = navigator.ml.createContext().tf;
  if (tf) {
    if (!(await tf.setBackend(backend))) {
      throw new Error(`Failed to set tf.js backend ${backend}.`);
    }
    await tf.ready();
    addAlert(
        `This sample is running on ` +
        `<a href='https://github.com/webmachinelearning/webnn-polyfill'>` +
        `WebNN-polyfill</a> with tf.js ${tf.version_core} ` +
        `<b>${tf.getBackend()}</b> backend.`, 'info');
  }
}

// Get url params
export function getUrlParams() {
  const params = new URLSearchParams(location.search);
  // Get 'numRuns' param to run inference multiple times
  let numRuns = params.get('numRuns');
  numRuns = numRuns === null ? 1 : parseInt(numRuns);
  if (numRuns < 1) {
    addAlert(`Ignore the url param: 'numRuns', its value must be >= 1.`);
    numRuns = 1;
  }

  // Get 'powerPreference' param to set WebNN's 'MLPowerPreference' option
  let powerPreference = params.get('powerPreference');
  const powerPreferences = ['default', 'high-performance', 'low-power'];

  if (powerPreference && !powerPreferences.includes(powerPreference)) {
    addAlert(`Ignore the url param: 'powerPreference', its value must be ` +
        `one of {'default', 'high-performance', 'low-power'}.`);
    powerPreference = null;
  }


  return [numRuns, powerPreference];
}
