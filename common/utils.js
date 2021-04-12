'use strict';

import {numpy} from './libs/numpy.js';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

export async function buildConstantByNpy(builder, url, isDepthwise = false) {
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
  let dimensions = npArray.shape;
  const type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  let typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
  const dataView = new DataView(npArray.data.buffer);
  const littleEndian = npArray.byteOrder === '<';
  for (let i = 0; i < sizeOfShape(dimensions); ++i) {
    typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
        i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
  }
  // TODO(Wanming): This is a workaround to transpose 'ihwo' to 'hwio',
  // and will be removed once 'ihwo' filterLayout is supported.
  if (isDepthwise) {
    const a = tf.tensor(typedArray, dimensions, type);
    const b = tf.transpose(a, [1, 2, 0, 3]);
    const buffer = await b.buffer();
    dimensions = b.shape;
    typedArray = buffer.values;
    tf.dispose();
  }
  return builder.constant({type, dimensions}, typedArray);
}

/**
 * This method is used to covert input element to tensor data.
 * @param {Object} inputElement, an object of HTML [<img> | <video>] element.
 * @param {!Object<string, *>} options, an object of options to process
 * input element.
 * @return {Object} tensor, an object of input tensor.
 */
export function getInputTensor(inputElement, options) {
  const inputDimensions = options.inputDimensions;
  const tensor = new Float32Array(
      inputDimensions.slice(1).reduce((a, b) => a * b));

  inputElement.width = inputElement.videoWidth ||
      inputElement.naturalWidth;
  inputElement.height = inputElement.videoHeight ||
      inputElement.naturalHeight;

  let [channels, height, width] = inputDimensions.slice(1);
  const mean = options.mean || [0, 0, 0, 0];
  const std = options.std || [1, 1, 1, 1];
  const normlizationFlag = options.norm || false;
  const nchwFlag = options.nchwFlag || false;
  const imageChannels = 4; // RGBA

  if (!nchwFlag) {
    [height, width, channels] = inputDimensions.slice(1);
  }
  const canvasElement = document.createElement('canvas');
  canvasElement.width = width;
  canvasElement.height = height;
  const canvasContext = canvasElement.getContext('2d');
  canvasContext.drawImage(inputElement, 0, 0, width, height);

  let pixels = canvasContext.getImageData(0, 0, width, height).data;

  if (normlizationFlag) {
    pixels = new Float32Array(pixels).map((p) => p / 255);
  }

  for (let c = 0; c < channels; ++c) {
    for (let h = 0; h < height; ++h) {
      for (let w = 0; w < width; ++w) {
        const value =
            pixels[h * width * imageChannels + w * imageChannels + c];
        if (nchwFlag) {
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
