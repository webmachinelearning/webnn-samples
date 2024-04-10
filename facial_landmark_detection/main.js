'use strict';

import {FaceLandmarkNhwc} from './face_landmark_nhwc.js';
import {FaceLandmarkNchw} from './face_landmark_nchw.js';
import {SsdMobilenetV2FaceNhwc} from './ssd_mobilenetv2_face_nhwc.js';
import {SsdMobilenetV2FaceNchw} from './ssd_mobilenetv2_face_nchw.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';
import * as SsdDecoder from '../common/libs/ssdDecoder.js';
import * as FaceLandmark from './libs/face_landmark_utils.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
let fdModelName = '';
const fldModelName = 'facelandmark';
let layout = 'nhwc';
let fdInstanceType = fdModelName + layout;
let fldInstanceType = fldModelName + layout;
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let fdInstance = null;
let fdInputOptions;
let fldInstance = null;
let fldInputOptions;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let fdOutputs;
let fldOutputs;
let deviceType = '';
let lastdeviceType = '';
let backend = '';
let lastBackend = '';
let stopRender = true;
let isRendering = false;
const disabledSelectors = ['#tabs > li', '.btn'];

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

$('#fdModelBtns .btn').on('change', async (e) => {
  if (inputType === 'camera') {
    await stopCamRender();
  }
  fdModelName = $(e.target).attr('id');
  await main();
});

// $('#layoutBtns .btn').on('change', async (e) => {
//   if (inputType === 'camera') {
//     await stopCamRender();
//   }
//   layout = $(e.target).attr('id');
//   await main();
// });

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
  const inputCanvas = utils.getVideoFrame(camElement);
  console.log('- Computing... ');
  const [totalComputeTime, strokedRects, keyPoints] =
      await predict(camElement);
  console.log(`  done in ${totalComputeTime} ms.`);
  computeTime = totalComputeTime;
  showPerfResult();
  await drawOutput(inputCanvas, strokedRects, keyPoints);
  $('#fps').text(`${(1000/totalComputeTime).toFixed(0)} FPS`);
  isRendering = false;
  if (!stopRender) {
    rafReq = requestAnimationFrame(renderCamStream);
  }
}

async function predict(inputElement) {
  const fdInputBuffer = utils.getInputTensor(inputElement, fdInputOptions);
  let totalComputeTime = 0;
  let start = performance.now();
  const results = await fdInstance.compute(fdInputBuffer, fdOutputs);
  totalComputeTime += performance.now() - start;
  fdOutputs = results.outputs;
  const strokedRects = [];
  const keyPoints = [];
  const height = inputElement.naturalHeight || inputElement.height;
  const width = inputElement.naturalWidth || inputElement.width;
  const fdOutputArrary = [];
  for (const output of Object.entries(fdOutputs)) {
    fdOutputArrary.push(output[1]);
  }
  const fdSsdOutputs = SsdDecoder.processSsdOutputTensor(
      fdOutputArrary, fdInputOptions, fdInstance.outputsInfo);

  const anchors = SsdDecoder.generateAnchors({});
  SsdDecoder.decodeOutputBoxTensor({}, fdSsdOutputs.outputBoxTensor, anchors);
  let [totalDetections, boxesList, scoresList] = SsdDecoder.nonMaxSuppression(
      {numClasses: 2},
      fdSsdOutputs.outputBoxTensor,
      fdSsdOutputs.outputClassScoresTensor);
  boxesList = SsdDecoder.cropSsdBox(
      inputElement, totalDetections, boxesList, fdInputOptions.margin);
  for (let i = 0; i < totalDetections; ++i) {
    let [ymin, xmin, ymax, xmax] = boxesList[i];
    ymin = Math.max(0, ymin) * height;
    xmin = Math.max(0, xmin) * width;
    ymax = Math.min(1, ymax) * height;
    xmax = Math.min(1, xmax) * width;
    const prob = 1 / (1 + Math.exp(-scoresList[i]));
    const rect = [xmin, ymin, xmax - xmin, ymax - ymin, prob];
    strokedRects.push(rect);
    const drawOptions= {
      sx: xmin,
      sy: ymin,
      sWidth: rect[2],
      sHeight: rect[3],
      dWidth: 128,
      dHeight: 128,
    };
    fldInputOptions.drawOptions = drawOptions;
    const fldInputBuffer = utils.getInputTensor(inputElement, fldInputOptions);
    start = performance.now();
    const results = await fldInstance.compute(fldInputBuffer, fldOutputs);
    totalComputeTime += performance.now() - start;
    fldOutputs = results.outputs;
    keyPoints.push(fldOutputs.output.slice());
  }
  return [totalComputeTime.toFixed(2), strokedRects, keyPoints];
}
async function drawOutput(inputElement, strokedRects, keyPoints) {
  const outputElement = document.getElementById('outputCanvas');
  $('#inferenceresult').show();

  const texts = strokedRects.map((r) => r[4].toFixed(2));
  SsdDecoder.drawFaceRectangles(
      inputElement, outputElement, strokedRects, texts);
  FaceLandmark.drawKeyPoints(
      inputElement, outputElement, keyPoints, strokedRects);
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
    'ssdmobilenetv2facenchw': new SsdMobilenetV2FaceNchw(),
    'ssdmobilenetv2facenhwc': new SsdMobilenetV2FaceNhwc(),
    'facelandmarknchw': new FaceLandmarkNchw(),
    'facelandmarknhwc': new FaceLandmarkNhwc(),
  };

  return netObject[type];
}

async function main() {
  try {
    if (fdModelName === '') return;
    [backend, deviceType] =
        $('input[name="backend"]:checked').attr('id').split('_');
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    const [numRuns, powerPreference, numThreads] = utils.getUrlParams();
    let start;
    // Only do load() and build() when model first time loads,
    // there's new model choosed, backend changed or device changed
    if (isFirstTimeLoad || fdInstanceType !== fdModelName + layout ||
        lastdeviceType != deviceType || lastBackend != backend) {
      if (lastdeviceType != deviceType || lastBackend != backend) {
        // Set backend and device
        await utils.setBackend(backend, deviceType);
        lastdeviceType = lastdeviceType != deviceType ?
                               deviceType : lastdeviceType;
        lastBackend = lastBackend != backend ? backend : lastBackend;
      }
      if (fldInstance !== null) {
        // Call dispose() to and avoid memory leak
        fldInstance.dispose();
      }
      if (fdInstance !== null) {
        // Call dispose() to and avoid memory leak
        fdInstance.dispose();
      }
      fdInstanceType = fdModelName + layout;
      fldInstanceType = fldModelName + layout;
      fdInstance = constructNetObject(fdInstanceType);
      fldInstance = constructNetObject(fldInstanceType);
      fdInputOptions = fdInstance.inputOptions;
      fldInputOptions = fldInstance.inputOptions;
      fdOutputs = {};
      for (const outputInfo of Object.entries(fdInstance.outputsInfo)) {
        fdOutputs[outputInfo[0]] =
            new Float32Array(utils.sizeOfShape(outputInfo[1]));
      }
      fldOutputs = {'output': new Float32Array(utils.sizeOfShape([1, 136]))};
      isFirstTimeLoad = false;
      console.log(`- Model name: ${fdModelName}, Model layout: ${layout} -`);
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
      const [fdOutputOperand, fldOutputOperand] = await Promise.all([
        fdInstance.load(contextOptions),
        fldInstance.load(contextOptions),
      ]);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      await Promise.all([
        fdInstance.build(fdOutputOperand),
        fldInstance.build(fldOutputOperand),
      ]);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const computeTimeArray = [];
      let strokedRects;
      let keyPoints;
      let medianComputeTime;
      console.log('- Computing... ');
      // Do warm up
      const fdResults = await fdInstance.compute(new Float32Array(
          utils.sizeOfShape(fdInputOptions.inputDimensions)), fdOutputs);
      const fldResults = await fldInstance.compute(new Float32Array(
          utils.sizeOfShape(fldInputOptions.inputDimensions)), fldOutputs);
      fdOutputs = fdResults.outputs;
      fldOutputs = fldResults.outputs;
      for (let i = 0; i < numRuns; i++) {
        [computeTime, strokedRects, keyPoints] = await predict(imgElement);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log('Face Detection model outputs: ', fdOutputs);
      console.log('Face Landmark model outputs: ', fldOutputs);
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(imgElement, strokedRects, keyPoints);
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
