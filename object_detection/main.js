'use strict';

import {TinyYoloV2Nchw} from './tiny_yolov2_nchw.js';
import {TinyYoloV2Nhwc} from './tiny_yolov2_nhwc.js';
import {SsdMobilenetV1Nchw} from './ssd_mobilenetv1_nchw.js';
import {SsdMobilenetV1Nhwc} from './ssd_mobilenetv1_nhwc.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';
import * as Yolo2Decoder from './libs/yolo2Decoder.js';
import * as SsdDecoder from '../common/libs/ssdDecoder.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelName = '';
let modelId = '';
let layout = 'nhwc';
let dataType = 'float32';
let instanceType = modelName + layout;
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let netInstance = null;
let labels = null;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let inputOptions;
let deviceType = '';
let lastdeviceType = '';
let backend = '';
let lastBackend = '';
let stopRender = true;
let isRendering = false;
const disabledSelectors = ['#tabs > li', '.btn'];
const modelIds = ['ssdmobilenetv1', 'tinyyolov2'];
const modelList = {
  'cpu': {
    'float32': modelIds,
  },
  'gpu': {
    'float32': modelIds,
    'float16': modelIds,
  },
  'npu': {
    'float16': ['ssdmobilenetv1'],
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
    $('#webnn_cpu').click();
  } else {
    console.log(utils.webNNNotSupportMessage());
    ui.addAlert(utils.webNNNotSupportMessageHTML());
  }
});

$('#backendBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }
  const backendId = $(e.target).attr('id');
  layout = utils.getDefaultLayout(backendId);
  [backend, deviceType] = backendId.split('_');
  // Only show the supported models for each deviceType. Now fp16 nchw models
  // are only supported on gpu/npu.
  if (backendId == 'webnn_gpu') {
    ui.handleBtnUI('#float16Label', false);
    ui.handleBtnUI('#float32Label', false);
    utils.displayAvailableModels(modelList, modelIds, deviceType, dataType);
  } else if (backendId == 'webnn_npu') {
    ui.handleBtnUI('#float16Label', false);
    ui.handleBtnUI('#float32Label', true);
    $('#float16').click();
    utils.displayAvailableModels(modelList, modelIds, deviceType, 'float16');
  } else {
    ui.handleBtnUI('#float16Label', true);
    ui.handleBtnUI('#float32Label', false);
    $('#float32').click();
    utils.displayAvailableModels(modelList, modelIds, deviceType, 'float32');
  }

  // Uncheck selected model
  if (modelId != '') {
    $(`#${modelId}`).parent().removeClass('active');
  }
});

$('#modelBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }

  modelId = $(e.target).attr('id');
  modelName = modelId;
  if (dataType == 'float16') {
    modelName += 'fp16';
  }

  await main();
});

$('#dataTypeBtns .btn').on('change', async (e) => {
  dataType = $(e.target).attr('id');
  utils.displayAvailableModels(modelList, modelIds, deviceType, dataType);
  // Uncheck selected model
  if (modelId != '') {
    $(`#${modelId}`).parent().removeClass('active');
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
  const results = await netInstance.compute(inputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(inputCanvas, results, labels);
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  isRendering = false;
  if (!stopRender) {
    rafReq = requestAnimationFrame(renderCamStream);
  }
}

async function drawOutput(inputElement, outputs, labels) {
  const outputElement = document.getElementById('outputCanvas');
  $('#inferenceresult').show();

  // Draw output for SSD Mobilenet V1 model
  if (modelName.includes('ssdmobilenetv1')) {
    const anchors = SsdDecoder.generateAnchors({});
    SsdDecoder.decodeOutputBoxTensor({}, outputs.boxes, anchors);
    let [totalDetections, boxesList, scoresList, classesList] =
        SsdDecoder.nonMaxSuppression({}, outputs.boxes, outputs.scores);
    boxesList = SsdDecoder.cropSsdBox(
        inputElement, totalDetections, boxesList, inputOptions.margin);
    SsdDecoder.drawBoxes(
        outputElement, totalDetections, inputElement,
        boxesList, scoresList, classesList, labels);
  } else {
    // Draw output for Tiny Yolo V2 model
    const outputBuffer = outputs.output;
    const decodeOut = Yolo2Decoder.decodeYOLOv2({numClasses: 20},
        outputBuffer, inputOptions.anchors);
    const boxes = Yolo2Decoder.getBoxes(decodeOut, inputOptions.margin);
    Yolo2Decoder.drawBoxes(inputElement, outputElement, boxes, labels);
  }
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

function constructNetObject(type) {
  const netObject = {
    'tinyyolov2nchw': new TinyYoloV2Nchw(),
    'tinyyolov2fp16nchw': new TinyYoloV2Nchw('float16'),
    'tinyyolov2nhwc': new TinyYoloV2Nhwc(),
    'ssdmobilenetv1nchw': new SsdMobilenetV1Nchw(),
    'ssdmobilenetv1fp16nchw': new SsdMobilenetV1Nchw('float16'),
    'ssdmobilenetv1nhwc': new SsdMobilenetV1Nhwc(),
  };

  return netObject[type];
}

async function main() {
  try {
    if (modelName === '') return;
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    let start;
    const [numRuns, powerPreference] = utils.getUrlParams();

    // Only do load() and build() when model first time loads,
    // there's new model choosed, backend changed or device changed
    if (isFirstTimeLoad || instanceType !== modelName + layout ||
        lastdeviceType != deviceType || lastBackend != backend) {
      if (lastdeviceType != deviceType || lastBackend != backend) {
        // Set backend and device
        lastdeviceType = lastdeviceType != deviceType ?
                               deviceType : lastdeviceType;
        lastBackend = lastBackend != backend ? backend : lastBackend;
      }
      instanceType = modelName + layout;
      netInstance = constructNetObject(instanceType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);

      isFirstTimeLoad = false;
      console.log(`- Model name: ${modelName}, Model layout: ${layout} -`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
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
      const results = await netInstance.compute(inputBuffer);

      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        await netInstance.compute(inputBuffer);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log('output: ', results);
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(imgElement, results, labels);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      stream = await utils.getMediaStream();
      camElement.srcObject = stream;
      stopRender = false;
      camElement.onloadeddata = await renderCamStream();
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
