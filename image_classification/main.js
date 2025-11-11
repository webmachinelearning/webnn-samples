'use strict';

import {ResNet50V1FP16Nchw} from './resnet50v1_fp16_nchw.js';
import {EfficientNetFP16Nchw} from './efficientnet_fp16_nchw.js';
import {MobileNetV2Nchw} from './mobilenet_nchw.js';
import {MobileNetV2Nhwc} from './mobilenet_nhwc.js';
import {MobileNetV2Uint8Nhwc} from './mobilenet_uint8_nhwc.js';
import {SqueezeNetNchw} from './squeezenet_nchw.js';
import {SqueezeNetNhwc} from './squeezenet_nhwc.js';
import {ResNet50V2Nchw} from './resnet50v2_nchw.js';
import {ResNet50V2Nhwc} from './resnet50v2_nhwc.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelName = '';
let layout = '';
let dataType = 'float32';
let instanceType = '';
let rafReq;
let inputType = 'image';
let netInstance = null;
let labels = null;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let inputOptions;
let deviceType = '';
let stopRender = true;
let isRendering = false;
const disabledSelectors = ['#tabs > li', '.btn'];
const modelIds = [
  'efficientnet',
  'mobilenet',
  'resnet50v1',
  'resnet50v2',
  'squeezenet',
];
const modelList = {
  'nhwc': {
    'float32': [
      'mobilenet',
      'squeezenet',
      'resnet50v2',
    ],
    'float16': [
      'mobilenet',
      'squeezenet',
      'resnet50v2',
    ],
    'uint8': [
      'mobilenet',
    ],
  },
  'nchw': {
    'float32': [
      'mobilenet',
      'squeezenet',
      'resnet50v2',
    ],
    'float16': modelIds,
  },
};

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

$(document).ready(async () => {
  $('.icdisplay').hide();
  if (await utils.isWebNN()) {
    $('#cpu').click();
  } else {
    console.log(utils.webNNNotSupportMessage());
    ui.addAlert(utils.webNNNotSupportMessageHTML());
  }
  layout = await utils.getDefaultLayout('cpu');
});

$('#deviceTypeBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }
  deviceType = $(e.target).attr('id');
  layout = await utils.getDefaultLayout(deviceType);
  const showUint8 = layout === 'nhwc' ? true : false;
  ui.handleBtnUI('#uint8Label', !showUint8);
  ui.handleBtnUI('#float16Label', false);
  // Only show the supported models for each deviceType.
  if (deviceType == 'npu') {
    ui.handleBtnUI('#float32Label', true);
    $('#float16').click();
  } else {
    ui.handleBtnUI('#float32Label', false);
    $('#float32').click();
  }

  utils.displayAvailableModels(modelList, modelIds, layout, dataType);
  // Uncheck selected model
  if (modelName != '') {
    $(`#${modelName}`).parent().removeClass('active');
  }
});

$('#modelBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }
  modelName = $(e.target).attr('id');

  await main();
});

// $('#layoutBtns .btn').on('change', async (e) => {
//   if (inputType === 'camera') {
//     await stopCamRender();
//   }
//   layout = $(e.target).attr('id');
//   await main();
// });

$('#dataTypeBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }

  dataType = $(e.target).attr('id');
  utils.displayAvailableModels(modelList, modelIds, layout, dataType);
  // Uncheck selected model
  if (modelName != '') {
    $(`#${modelName}`).parent().removeClass('active');
  }
});


// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') {
    await stopCamRender();
  } else {
    return;
  }
  inputType = 'image';
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
  if (inputType == 'camera') return;
  inputType = 'camera';
  $('.shoulddisplay').hide();
  await main();
});

function stopCamRender() {
  stopRender = true;
  utils.stopCameraStream(rafReq, stream);
  return new Promise((resolve) => {
    // if the rendering is not stopped, check it every 100ms
    setInterval(() => {
      // resolve when the rendering is stopped
      if (!isRendering) {
        resolve();
      }
    }, 100);
  });
}

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  if (!stream.active || stopRender) return;
  // If the video element's readyState is 0, the video's width and height are 0.
  // So check the readState here to make sure it is greater than 0.
  if (camElement.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  isRendering = true;
  const inputBuffer = utils.getInputTensor(camElement, inputOptions);
  const inputCanvas = utils.getVideoFrame(camElement);
  console.log('- Computing... ');
  const start = performance.now();
  const outputBuffer = await netInstance.compute(inputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  drawInput(inputCanvas, 'camInCanvas');
  showPerfResult();
  await drawOutput(outputBuffer, labels);
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  isRendering = false;
  if (!stopRender) {
    rafReq = requestAnimationFrame(renderCamStream);
  }
}

// Get top 3 classes of labels from output buffer
function getTopClasses(buffer, labels) {
  const probs = Array.from(buffer);
  const indexes = probs.map((prob, index) => [prob, index]);
  const sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {
      return 0;
    }
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();
  const classes = [];

  for (let i = 0; i < 3; ++i) {
    const prob = sorted[i][0];
    const index = sorted[i][1];
    const c = {
      label: labels[index],
      prob: (prob * 100).toFixed(2),
    };
    classes.push(c);
  }

  return classes;
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

async function drawOutput(outputBuffer, labels) {
  const labelClasses = getTopClasses(outputBuffer, labels);

  $('#inferenceresult').show();
  labelClasses.forEach((c, i) => {
    const labelElement = document.getElementById(`label${i}`);
    const probElement = document.getElementById(`prob${i}`);
    labelElement.innerHTML = `${c.label}`;
    probElement.innerHTML = `${c.prob}%`;
  });
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

function constructNetObject(modelName, layout, dataType) {
  switch (modelName) {
    case 'efficientnet':
      if (layout == 'nchw' && dataType == 'float16') {
        return new EfficientNetFP16Nchw();
      }
      break;
    case 'mobilenet':
      if (layout == 'nhwc' && dataType == 'uint8') {
        return new MobileNetV2Uint8Nhwc();
      } else if (dataType != 'uint8') {
        return layout == 'nhwc' ?
            new MobileNetV2Nhwc(dataType) : new MobileNetV2Nchw(dataType);
      }
      break;
    case 'resnet50v1':
      if (layout == 'nchw' && dataType == 'float16') {
        return new ResNet50V1FP16Nchw();
      }
      break;
    case 'resnet50v2':
      if (dataType != 'uint8') {
        return layout == 'nhwc' ?
            new ResNet50V2Nhwc(dataType) : new ResNet50V2Nchw(dataType);
      }
      break;
    case 'squeezenet':
      if (dataType != 'uint8') {
        return layout == 'nhwc' ?
            new SqueezeNetNhwc() : new SqueezeNetNchw();
      }
      break;
  }

  throw new Error(`Unknown model, name: ${modelName}, layout: ${layout}, ` +
     `dataType: ${dataType}`);
}

async function main() {
  try {
    if (modelName === '') return;
    ui.handleClick(disabledSelectors, true);
    if (instanceType == '') $('#hint').hide();
    let start;
    const [numRuns, powerPreference] = utils.getUrlParams();

    // Only do load() and build() when model first time loads,
    // there's new model choosed
    if (instanceType !== modelName + dataType + layout + deviceType) {
      instanceType = modelName + dataType + layout + deviceType;
      netInstance = constructNetObject(modelName, layout, dataType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);
      console.log(
          `- Model: ${modelName} - ${layout} - ${dataType} - ${deviceType}`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights...');
      const contextOptions = {deviceType};
      if (powerPreference) {
        contextOptions['powerPreference'] = powerPreference;
      }
      start = performance.now();
      const outputOperand = await netInstance.load(contextOptions);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      await netInstance.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;

      // Do warm up
      let outputBuffer = await netInstance.compute(inputBuffer);

      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        outputBuffer = await netInstance.compute(inputBuffer);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log('outputBuffer: ', outputBuffer);
      await ui.showProgressComponent('done', 'done', 'done');
      ui.readyShowResultComponents();
      drawInput(imgElement, 'inputCanvas');
      await drawOutput(outputBuffer, labels);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      stream = await utils.getMediaStream();
      camElement.srcObject = stream;
      stopRender = false;
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
