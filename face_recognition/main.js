'use strict';

import {FaceNetNhwc} from './facenet_nhwc.js';
import {FaceNetNchw} from './facenet_nchw.js';
import {SsdMobilenetV2FaceNhwc} from '../facial_landmark_detection/ssd_mobilenetv2_face_nhwc.js';
import {SsdMobilenetV2FaceNchw} from '../facial_landmark_detection/ssd_mobilenetv2_face_nchw.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';
import * as SsdDecoder from '../common/libs/ssdDecoder.js';
import * as FaceRecognitionUtils from './libs/face_recognition_utils.js';

const searchImgElem = document.getElementById('searchImage');
const searchCanvasShowElem = document.getElementById('searchCanvasShow');
const searchCanvasCamShowElem = document.getElementById('cameraShow');
const targetImgElem = document.getElementById('targetImage');
const camElem = document.getElementById('camElement');
let targetEmbeddings = null;
let searchEmbeddings = null;
let fdModelName = '';
const frModelName = 'facenet';
let layout = 'nhwc';
let fdInstanceType = fdModelName + layout;
let frInstanceType = frModelName + layout;
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let fdInstance = null;
let fdInputOptions;
let frInstance = null;
let frInputOptions;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let fdOutputs;
let frOutputs;
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
    await ui.showProgressComponent('current', 'pending', 'pending');
    await stopCamRender();
    // Set timeout to leave more time to make sure searchEmbeddings
    // is clear after switching from camera tab to image tab
    await new Promise((resolve) => {
      setTimeout(() => {
        searchEmbeddings = null;
        resolve();
      }, 1000);
    });
  } else {
    return;
  }
  inputType = 'image';
  searchEmbeddings = null;
  await main();
});

$('#targetImgFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#targetImage').removeAttr('height');
    $('#targetImage').removeAttr('width');
    targetImgElem.src = URL.createObjectURL(files[0]);
  }
});

$('#targetImage').on('load', async () => {
  targetEmbeddings = null;
  if (inputType === 'image') {
    await main();
  }
});

$('#searchImgFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#searchImage').removeAttr('height');
    $('#searchImage').removeAttr('width');
    searchImgElem.src = URL.createObjectURL(files[0]);
  }
});

$('#searchImage').on('load', async () => {
  searchEmbeddings = null;
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
  if (camElem.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  isRendering = true;
  // Clear search embeddings for each frame
  searchEmbeddings = null;
  const inputCanvas = utils.getVideoFrame(camElem);
  console.log('- Computing... ');
  await predict(targetImgElem, camElem);
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(inputCanvas, searchCanvasCamShowElem);
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  isRendering = false;
  if (!stopRender) {
    rafReq = requestAnimationFrame(renderCamStream);
  }
}

async function getEmbeddings(inputElem) {
  const fdInputBuffer = utils.getInputTensor(inputElem, fdInputOptions);
  let totalComputeTime = 0;
  let start = performance.now();
  const results = await fdInstance.compute(fdInputBuffer, fdOutputs);
  totalComputeTime = performance.now() - start;
  fdOutputs = results.outputs;
  const strokedRects = [];
  const embeddings = [];
  const height = inputElem.naturalHeight || inputElem.height;
  const width = inputElem.naturalWidth || inputElem.width;
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
      inputElem, totalDetections, boxesList, fdInputOptions.margin);
  for (let i = 0; i < totalDetections; ++i) {
    let [ymin, xmin, ymax, xmax] = boxesList[i];
    ymin = Math.max(0, ymin) * height;
    xmin = Math.max(0, xmin) * width;
    ymax = Math.min(1, ymax) * height;
    xmax = Math.min(1, xmax) * width;
    const prob = 1 / (1 + Math.exp(-scoresList[i]));
    const rect = [xmin, ymin, xmax - xmin, ymax - ymin, prob];
    strokedRects.push(rect);
    const drawOptions = {
      sx: xmin,
      sy: ymin,
      sWidth: rect[2],
      sHeight: rect[3],
      dWidth: 160,
      dHeight: 160,
    };
    frInputOptions.drawOptions = drawOptions;
    const frInputBuffer = utils.getInputTensor(inputElem, frInputOptions);
    start = performance.now();
    const results = await frInstance.compute(frInputBuffer, frOutputs);
    totalComputeTime += performance.now() - start;
    frOutputs = results.outputs;
    const [...normEmbedding] = Float32Array.from(frOutputs.output);
    embeddings.push(normEmbedding);
  }
  return {computeTime: totalComputeTime, strokedRects, embeddings};
}

async function predict(targetElem, searchElem) {
  let flag1 = false;
  let flag2 = false;
  if (targetEmbeddings == null) {
    targetEmbeddings = await getEmbeddings(targetElem);
    flag1 = true;
  }
  if (searchEmbeddings == null) {
    searchEmbeddings = await getEmbeddings(searchElem);
    flag2 = true;
  }
  if (flag1 && flag2) {
    computeTime = targetEmbeddings.computeTime + searchEmbeddings.computeTime;
  } else if (flag1 && !flag2) {
    computeTime = targetEmbeddings.computeTime;
  } else if (!flag1 && flag2) {
    computeTime = searchEmbeddings.computeTime;
  }
}

async function drawOutput(searchElem, searchCanvasShowElem) {
  $('#inferenceresult').show();

  const targetTextClasses = [];
  for (let i = 0; i < targetEmbeddings.embeddings.length; i++) {
    targetTextClasses.push(i + 1);
  }

  const targetCanvasShowElem = document.getElementById('targetCanvasShow');
  SsdDecoder.drawFaceRectangles(targetImgElem,
      targetCanvasShowElem,
      targetEmbeddings.strokedRects,
      targetTextClasses, 300);
  const searchTextClasses = FaceRecognitionUtils.getFRClass(
      targetEmbeddings.embeddings, searchEmbeddings.embeddings,
      frInstance.postOptions);
  SsdDecoder.drawFaceRectangles(searchElem,
      searchCanvasShowElem,
      searchEmbeddings.strokedRects,
      searchTextClasses, 300);
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime.toFixed(2)} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime.toFixed(2)} ms`);
  }
}

function constructNetObject(type) {
  const netObject = {
    'ssdmobilenetv2facenchw': new SsdMobilenetV2FaceNchw(),
    'ssdmobilenetv2facenhwc': new SsdMobilenetV2FaceNhwc(),
    'facenetnchw': new FaceNetNchw(),
    'facenetnhwc': new FaceNetNhwc(),
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
      if (frInstance !== null) {
        // Call dispose() to and avoid memory leak
        frInstance.dispose();
      }
      if (fdInstance !== null) {
        // Call dispose() to and avoid memory leak
        fdInstance.dispose();
      }
      fdInstanceType = fdModelName + layout;
      frInstanceType = frModelName + layout;
      fdInstance = constructNetObject(fdInstanceType);
      frInstance = constructNetObject(frInstanceType);
      fdInputOptions = fdInstance.inputOptions;
      frInputOptions = frInstance.inputOptions;
      fdOutputs = {};
      for (const outputInfo of Object.entries(fdInstance.outputsInfo)) {
        fdOutputs[outputInfo[0]] =
            new Float32Array(utils.sizeOfShape(outputInfo[1]));
      }
      frOutputs = {'output': new Float32Array(utils.sizeOfShape([1, 512]))};
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
      const [fdOutputOperand, frOutputOperand] = await Promise.all([
        fdInstance.load(contextOptions),
        frInstance.load(contextOptions),
      ]);

      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      await Promise.all([
        fdInstance.build(fdOutputOperand),
        frInstance.build(frOutputOperand),
      ]);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const computeTimeArray = [];
      let medianComputeTime;
      console.log('- Computing... ');
      // Do warm up
      const fdResults = await fdInstance.compute(new Float32Array(
          utils.sizeOfShape(fdInputOptions.inputDimensions)), fdOutputs);
      const frResults = await frInstance.compute(new Float32Array(
          utils.sizeOfShape(frInputOptions.inputDimensions)), frOutputs);
      fdOutputs = fdResults.outputs;
      frOutputs = frResults.outputs;
      for (let i = 0; i < numRuns; i++) {
        if (numRuns > 1) {
          // clear all predicted embeddings for benckmarking
          targetEmbeddings = null;
          searchEmbeddings = null;
        }
        await predict(targetImgElem, searchImgElem);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(computeTime);
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        console.log(
            `  median compute time: ${medianComputeTime.toFixed(2)} ms`);
      }
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(searchImgElem, searchCanvasShowElem);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      stream = await utils.getMediaStream();
      camElem.srcObject = stream;
      stopRender = false;
      camElem.onloadeddata = await renderCamStream();
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
