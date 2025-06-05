'use strict';

import * as utils from '../common/utils.js';
import {buildWebGL2Pipeline} from './lib/webgl2/webgl2Pipeline.js';
import * as ui from '../common/ui.js';
import {SelfieSegmentationGeneral} from './selfie_segmentation_general.js';
import {SelfieSegmentationLandscape} from './selfie_segmentation_landscape.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
const outputCanvas = document.getElementById('outputCanvas');
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let stream = null;
let netInstance;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let outputBuffer;
let modelChanged = false;
let backgroundImageSource = document.getElementById('00-img');
let backgroundType = 'image'; // 'none', 'blur', 'image'
let deviceType = 'cpu';
let resolutionType = '';
let dataType = 'float16';
const inputOptions = {
  mean: [127.5, 127.5, 127.5],
  std: [127.5, 127.5, 127.5],
  scaledFlag: false,
  inputResolution: [256, 144],
};

const disabledSelectors = ['#tabs > li', '.btn'];

$(document).ready(async () => {
  $('.icdisplay').hide();
  if (await utils.isWebNN()) {
    $('#cpu').click();
  } else {
    console.log(utils.webNNNotSupportMessage());
    ui.addAlert(utils.webNNNotSupportMessageHTML());
  }
});

$('#backendBtns .btn').on('change', async (e) => {
  modelChanged = true;
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  deviceType = $(e.target).attr('id');
  await main();
});

$('#dataTypeBtns .btn').on('change', async (e) => {
  modelChanged = true;
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  dataType = $(e.target).attr('id');
  await main();
});

$('#resolutionType').on('change', async (e) => {
  modelChanged = true;
  resolutionType = $(e.target).attr('id');
  if (resolutionType === 'general') {
    inputOptions.inputResolution = [256, 256];
  } else {
    inputOptions.inputResolution = [256, 144];
  }
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  inputType = 'image';
  $('#pickimage').show();
  $('.shoulddisplay').hide();
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
  await main();
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('#pickimage').hide();
  $('.shoulddisplay').hide();
  await main();
});

$('#gallery .gallery-item').click(async (e) => {
  if ($(e.target).attr('id') == backgroundType) return;
  $('#gallery .gallery-item').removeClass('hl');
  $(e.target).parent().addClass('hl');
  const backgroundTypeId = $(e.target).attr('id');
  backgroundImageSource = document.getElementById(backgroundTypeId);
  if (backgroundTypeId === 'no-img') {
    backgroundType = 'none';
  } else if (backgroundTypeId === 'blur-img') {
    backgroundType = 'blur';
  } else {
    backgroundType = 'image';
  }
  const srcElement = inputType == 'image' ? imgElement : camElement;
  await drawOutput(outputBuffer, srcElement);
});

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  if (!stream.active) return;
  // If the video element's readyState is 0, the video's width and height are 0.
  // So check the readState here to make sure it is greater than 0.
  if (camElement.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  const inputCanvas = utils.getVideoFrame(camElement);
  let inputBuffer = utils.getInputTensor(camElement, inputOptions);
  if (dataType == 'float16') {
    inputBuffer = Float16Array.from(inputBuffer);
  }

  const start = performance.now();
  outputBuffer = await netInstance.compute(inputBuffer);
  if (dataType == 'float16') {
    outputBuffer = Float32Array.from(outputBuffer);
  }
  computeTime = performance.now() - start;
  console.log(`  done in ${computeTime.toFixed(2)} ms.`);

  showPerfResult();
  await drawOutput(outputBuffer, inputCanvas);
  $('#fps').text(`${(1000 / computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

async function drawOutput(outputBuffer, srcElement) {
  outputCanvas.width = srcElement.width;
  outputCanvas.height = srcElement.height;
  const pipeline = buildWebGL2Pipeline(
      srcElement,
      backgroundImageSource,
      backgroundType,
      inputOptions.inputResolution,
      outputCanvas,
      outputBuffer,
  );
  const postProcessingConfig = {
    smoothSegmentationMask: true,
    jointBilateralFilter: {sigmaSpace: 1, sigmaColor: 0.1},
    coverage: [0.5, 0.75],
    lightWrapping: 0.3,
    blendMode: 'screen',
  };
  pipeline.updatePostProcessingConfig(postProcessingConfig);
  await pipeline.render();
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime.toFixed(2)} ms`);
  $('#buildTime').html(`${buildTime.toFixed(2)} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime.toFixed(2)} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime.toFixed(2)} ms`);
  }
}

export async function main() {
  try {
    if (resolutionType === '') return;
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    const numRuns = utils.getUrlParams()[0];
    // Only do load() and build() when model first time loads,
    // there's new model choosed or device changed
    if (isFirstTimeLoad || modelChanged) {
      modelChanged = false;
      isFirstTimeLoad = false;
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      let start = performance.now();
      netInstance =
        resolutionType == 'landscape' ?
          new SelfieSegmentationLandscape(deviceType, dataType) :
          new SelfieSegmentationGeneral(deviceType, dataType);
      const graph = await netInstance.load({deviceType});
      inputOptions.inputLayout = netInstance.layout;
      inputOptions.inputShape = netInstance.inputShape;
      console.log(`- Loading WebNN model: [${resolutionType}]
 deviceType: [${deviceType}] dataType: [${dataType}]
 preferredLayout: [${netInstance.layout}]`);
      loadTime = performance.now() - start;
      console.log(`  done in ${loadTime.toFixed(2)} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      await netInstance.build(graph);
      buildTime = performance.now() - start;
      console.log(`  done in ${buildTime.toFixed(2)} ms.`);
    }

    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      let inputBuffer = utils.getInputTensor(imgElement, inputOptions);
      if (dataType == 'float16') {
        inputBuffer = Float16Array.from(inputBuffer);
      }
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;

      // Do warm up
      outputBuffer = await netInstance.compute(inputBuffer);
      if (dataType == 'float16') {
        outputBuffer = Float32Array.from(outputBuffer);
      }
      for (let i = 0; i < numRuns; i++) {
        const start = performance.now();
        await netInstance.compute(inputBuffer);
        computeTime = performance.now() - start;
        console.log(`  compute time ${i + 1}: ${computeTime.toFixed(2)} ms`);
        computeTimeArray.push(computeTime);
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        console.log(
            `  median compute time: ${medianComputeTime.toFixed(2)} ms`);
      }
      console.log('output: ', outputBuffer);
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(outputBuffer, imgElement);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      stream = await utils.getMediaStream({
        width: inputOptions.inputResolution[0],
        height: inputOptions.inputResolution[1],
      });
      camElement.srcObject = stream;
      camElement.onloadedmediadata = await renderCamStream();
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').show();
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
