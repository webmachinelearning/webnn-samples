'use strict';

import {FastStyleTransferNet} from './fast_style_transfer_net.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/content-images/travelspace.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelId = 'starry-night';
let isFirstTimeLoad = true;
let isModelChanged = false;
let rafReq;
let inputType = 'image';
let fastStyleTransferNet;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let outputBuffer;
let devicePreference = 'gpu';
let lastDevicePreference = '';
const disabledSelectors = [
  '#tabs > li',
  '#gallery',
  '#gallery > div > img',
  '.btn',
];

$(document).ready(() => {
  $('.icdisplay').hide();
  $('.badge').html(modelId);
});

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  if (stream !== null) stopCamera();
  inputType = 'image';
  $('.shoulddisplay').hide();
  await main();
});

$('#gallery .gallery-image').hover((e) => {
  const id = $(e.target).attr('id');
  const modelName = $('#' + id).attr('title');
  $('.badge').html(modelName);
}, () => {
  const modelName = $(`#${modelId}`).attr('title');
  $('.badge').html(modelName);
});

// Click trigger to do inference with switched <img> element
$('#gallery .gallery-item').click(async (e) => {
  const newModelId = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  if (newModelId !== modelId) {
    isModelChanged = true;
    modelId = newModelId;
    const modelName = $(`#${modelId}`).attr('title');
    $('.badge').html(modelName);
    $('#gallery .gallery-item').removeClass('hl');
    $(e.target).parent().addClass('hl');
  }
  await main();
});

$('#imageFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

$('#feedElement').on('load', async () => {
  if (!isFirstTimeLoad) {
    await main();
  }
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('.shoulddisplay').hide();
  await main();
});

async function getMediaStream() {
  // Support 'user' facing mode at present
  const constraints = {audio: false, video: {facingMode: 'user'}};
  stream = await navigator.mediaDevices.getUserMedia(constraints);
}

function stopCamera() {
  stream.getTracks().forEach((track) => {
    if (track.readyState === 'live' && track.kind === 'video') {
      track.stop();
    }
  });
}

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  // If the video element's readyState is 0, the video's width and height are 0.
  // So check the readState here to make sure it is greater than 0.
  if (camElement.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  const inputBuffer =
      utils.getInputTensor(camElement, fastStyleTransferNet.inputOptions);
  const inputCanvas = utils.getVideoFrame(camElement);
  console.log('- Computing... ');
  const start = performance.now();
  fastStyleTransferNet.compute(inputBuffer, outputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  camElement.width = camElement.videoWidth;
  camElement.height = camElement.videoHeight;
  drawInput(inputCanvas, 'camInCanvas');
  showPerfResult();
  drawOutput('camInCanvas', 'camOutCanvas');
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

function drawInput(srcElement, canvasId) {
  const inputCanvas = document.getElementById(canvasId);
  const resizeRatio = Math.max(
      Math.max(srcElement.width / maxWidth, srcElement.height / maxHeight), 1);
  const scaledWidth = Math.floor(srcElement.width / resizeRatio);
  const scaledHeight = Math.floor(srcElement.height / resizeRatio);
  inputCanvas.height = scaledHeight;
  inputCanvas.width = scaledWidth;
  const ctx = inputCanvas.getContext('2d');
  ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);
}

function drawOutput(inCanvasId, outCanvasId) {
  const outputSize = fastStyleTransferNet.outputDimensions;
  const height = outputSize[2];
  const width = outputSize[3];
  const mean = [1, 1, 1, 1];
  const offset = [0, 0, 0, 0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const a = 255;

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const r = outputBuffer[i] * mean[0] + offset[0];
    const g = outputBuffer[i + height * width] * mean[1] + offset[1];
    const b = outputBuffer[i + height * width * 2] * mean[2] + offset[2];
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }

  const imageData = new ImageData(bytes, width, height);
  const outCanvas = document.createElement('canvas');
  const outCtx = outCanvas.getContext('2d');
  outCanvas.width = width;
  outCanvas.height = height;
  outCtx.putImageData(imageData, 0, 0, 0, 0, outCanvas.width, outCanvas.height);

  const inputCanvas = document.getElementById(inCanvasId);
  const outputCanvas = document.getElementById(outCanvasId);
  outputCanvas.width = inputCanvas.width;
  outputCanvas.height = inputCanvas.height;
  const ctx = outputCanvas.getContext('2d');
  ctx.drawImage(outCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime} ms`);
  }
}

export async function main() {
  try {
    ui.handleClick(disabledSelectors, true);
    let start;
    // Set 'numRuns' param to run inference multiple times
    const params = new URLSearchParams(location.search);
    let numRuns = params.get('numRuns');
    numRuns = numRuns === null ? 1 : parseInt(numRuns);

    if (numRuns < 1) {
      ui.addAlert('The value of param numRuns must be greater than or equal' +
          ' to 1.');
      return;
    }
    // Only do load() and build() when model first time loads,
    // there's new model choosed, and device backend changed
    if (isFirstTimeLoad || isModelChanged ||
        lastDevicePreference != devicePreference) {
      if (lastDevicePreference != devicePreference) {
        // Set polyfill backend
        await utils.setPolyfillBackend(devicePreference);
        lastDevicePreference = devicePreference;
      }
      if (fastStyleTransferNet !== undefined) {
        // Call dispose() to and avoid memory leak
        fastStyleTransferNet.dispose();
      }
      fastStyleTransferNet = new FastStyleTransferNet();
      outputBuffer = new Float32Array(
          utils.sizeOfShape(fastStyleTransferNet.outputDimensions));
      isFirstTimeLoad = false;
      isModelChanged = false;
      console.log(`- Model ID: ${modelId} -`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      start = performance.now();
      const outputOperand =
          await fastStyleTransferNet.load(devicePreference, modelId);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      fastStyleTransferNet.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer =
          utils.getInputTensor(imgElement, fastStyleTransferNet.inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;
      if (numRuns > 1) {
        // Do warm up
        fastStyleTransferNet.compute(inputBuffer, outputBuffer);
      }
      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        fastStyleTransferNet.compute(inputBuffer, outputBuffer);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      await ui.showProgressComponent('done', 'done', 'done');
      ui.readyShowResultComponents();
      drawInput(imgElement, 'inputCanvas');
      drawOutput('inputCanvas', 'outputCanvas');
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
      camElement.onloadeddata = await renderCamStream();
      await ui.showProgressComponent('done', 'done', 'done');
      ui.readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
  } catch (error) {
    console.log(error);
    ui.addAlert(error.message);
  }
  ui.handleClick(disabledSelectors, false);
}
