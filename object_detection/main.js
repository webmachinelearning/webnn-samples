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
let layout = 'nhwc';
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
let outputs;
let deviceType = '';
let lastdeviceType = '';
let backend = '';
let lastBackend = '';
let stopRender = true;
let isRendering = false;
const disabledSelectors = ['#tabs > li', '.btn'];

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
    $('#polyfill_cpu').click();
  }
});

$('#backendBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }
  layout = utils.getDefaultLayout($(e.target).attr('id'));
  await main();
});

$('#modelBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }
  modelName = $(e.target).attr('id');
  await main();
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
  const results = await netInstance.compute(inputBuffer, outputs);
  outputs = results.outputs;
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(inputCanvas, outputs, labels);
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
  if (modelName === 'ssdmobilenetv1') {
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
    // Transpose 'nchw' output to 'nhwc' for postprocessing
    let outputBuffer = outputs.output;
    if (layout === 'nchw') {
      outputBuffer = tf.tidy(() => {
        const a =
            tf.tensor(outputBuffer, netInstance.outputDimensions, 'float32');
        const b = tf.transpose(a, [0, 2, 3, 1]);
        return b.dataSync();
      });
    }
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
    'tinyyolov2nhwc': new TinyYoloV2Nhwc(),
    'ssdmobilenetv1nchw': new SsdMobilenetV1Nchw(),
    'ssdmobilenetv1nhwc': new SsdMobilenetV1Nhwc(),
  };

  return netObject[type];
}

async function main() {
  try {
    if (modelName === '') return;
    [backend, deviceType] =
        $('input[name="backend"]:checked').attr('id').split('_');
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    let start;
    const [numRuns, powerPreference, numThreads] = utils.getUrlParams();

    // Only do load() and build() when model first time loads,
    // there's new model choosed, backend changed or device changed
    if (isFirstTimeLoad || instanceType !== modelName + layout ||
        lastdeviceType != deviceType || lastBackend != backend) {
      if (lastdeviceType != deviceType || lastBackend != backend) {
        // Set backend and device
        await utils.setBackend(backend, deviceType);
        lastdeviceType = lastdeviceType != deviceType ?
                               deviceType : lastdeviceType;
        lastBackend = lastBackend != backend ? backend : lastBackend;
      }
      if (netInstance !== null) {
        // Call dispose() to and avoid memory leak
        netInstance.dispose();
      }
      instanceType = modelName + layout;
      netInstance = constructNetObject(instanceType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);
      if (modelName === 'tinyyolov2') {
        outputs = {
          'output': new Float32Array(
              utils.sizeOfShape(netInstance.outputDimensions)),
        };
      } else {
        outputs = {
          'boxes': new Float32Array(utils.sizeOfShape([1, 1917, 1, 4])),
          'scores': new Float32Array(utils.sizeOfShape([1, 1917, 91])),
        };
      }
      isFirstTimeLoad = false;
      console.log(`- Model name: ${modelName}, Model layout: ${layout} -`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      const contextOptions = {deviceType};
      if (powerPreference) {
        contextOptions['powerPreference'] = powerPreference;
      }
      if (numThreads) {
        contextOptions['numThreads'] = numThreads;
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
      let results = await netInstance.compute(inputBuffer, outputs);

      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        results = await netInstance.compute(
            results.inputs.input, results.outputs);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      outputs = results.outputs;
      console.log('output: ', outputs);
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(imgElement, outputs, labels);
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
