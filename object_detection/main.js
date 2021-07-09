'use strict';

import {TinyYoloV2Nchw} from './tiny_yolov2_nchw.js';
import {TinyYoloV2Nhwc} from './tiny_yolov2_nhwc.js';
import {SsdMobilenetV1Nchw} from './ssd_mobilenetv1_nchw.js';
import {SsdMobilenetV1Nhwc} from './ssd_mobilenetv1_nhwc.js';
import {showProgressComponent, readyShowResultComponents} from '../common/ui.js';
import {getInputTensor, getMedianValue, sizeOfShape} from '../common/utils.js';
import * as Yolo2Decoder from './libs/yolo2Decoder.js';
import * as SsdDecoder from './libs/ssdDecoder.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelName = 'tinyyolov2';
let layout = 'nchw';
let instanceType = modelName + layout;
let shouldStopFrame = false;
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

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

$(document).ready(() => {
  $('.icdisplay').hide();
});

$('#modelBtns .btn').on('change', async (e) => {
  modelName = $(e.target).attr('id');
  shouldStopFrame = true;
  await main();
});

$('#layoutBtns .btn').on('change', async (e) => {
  layout = $(e.target).attr('id');
  shouldStopFrame = true;
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  shouldStopFrame = true;
  if (stream !== null) {
    stopCamera();
  }
  inputType = 'image';
  $('.shoulddisplay').hide();
  await main();
});

$('#imageFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#feedElement').on('load', async () => {
      await main();
    });
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
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
  const inputBuffer = getInputTensor(camElement, inputOptions);
  console.log('- Computing... ');
  const start = performance.now();
  netInstance.compute(inputBuffer, outputs);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  camElement.width = camElement.videoWidth;
  camElement.height = camElement.videoHeight;
  showPerfResult();
  await drawOutput(camElement, outputs, labels);
  if (!shouldStopFrame) {
    requestAnimationFrame(renderCamStream);
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
      const tf = navigator.ml.createContext().tf;
      const a =
          tf.tensor(outputBuffer, netInstance.outputDimensions, 'float32');
      const b = tf.transpose(a, [0, 2, 3, 1]);
      const buffer = await b.buffer();
      tf.dispose();
      outputBuffer = buffer.values;
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

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

export async function main() {
  try {
    $('input[type="radio"]').attr('disabled', true);
    let start;
    // Set 'numRuns' param to run inference multiple times
    const params = new URLSearchParams(location.search);
    let numRuns = params.get('numRuns');
    numRuns = numRuns === null ? 1 : parseInt(numRuns);

    if (numRuns < 1) {
      addWarning('The value of param numRuns must be greater than or equal' +
          ' to 1.');
      return;
    }
    // Only do load() and build() when model first time loads and
    // there's new model choosed
    if (isFirstTimeLoad || instanceType !== modelName + layout) {
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
          'output': new Float32Array(sizeOfShape(netInstance.outputDimensions)),
        };
      } else {
        outputs = {
          'boxes': new Float32Array(sizeOfShape([1, 1917, 1, 4])),
          'scores': new Float32Array(sizeOfShape([1, 1917, 91])),
        };
      }
      isFirstTimeLoad = false;
      console.log(`- Model name: ${modelName}, Model layout: ${layout} -`);
      // UI shows model loading progress
      await showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      start = performance.now();
      const outputOperand = await netInstance.load();
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      netInstance.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer = getInputTensor(imgElement, inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;
      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        netInstance.compute(inputBuffer, outputs);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log('output: ', outputs);
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
      await drawOutput(imgElement, outputs, labels);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
      shouldStopFrame = false;
      camElement.onloadedmediadata = await renderCamStream();
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
    $('input[type="radio"]').attr('disabled', false);
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
