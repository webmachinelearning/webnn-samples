'use strict';

import * as utils from '../common/utils.js';
import {buildWebGL2Pipeline} from './lib/webgl2/webgl2Pipeline.js';
import * as ui from '../common/ui.js';
const worker = new Worker('./builtin_delegate_worker.js');

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
const outputCanvas = document.getElementById('outputCanvas');
let modelName = '';
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let stream = null;
let loadTime = 0;
let computeTime = 0;
let outputBuffer;
let modelChanged = false;
let backgroundImageSource = document.getElementById('00-img');
let backgroundType = 'img'; // 'none', 'blur', 'image'
const inputOptions = {
  mean: [127.5, 127.5, 127.5],
  std: [127.5, 127.5, 127.5],
  scaledFlag: false,
  inputLayout: 'nhwc',
};
const modelConfigs = {
  'selfie_segmentation': {
    inputDimensions: [1, 256, 256, 3],
    inputResolution: [256, 256],
    modelPath: 'https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite',
  },
  'selfie_segmentation_landscape': {
    inputDimensions: [1, 144, 256, 3],
    inputResolution: [256, 144],
    modelPath: 'https://storage.googleapis.com/mediapipe-assets/selfie_segmentation_landscape.tflite',
  },
  'deeplabv3': {
    inputDimensions: [1, 257, 257, 3],
    inputResolution: [257, 257],
    modelPath: 'https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite',
  },
};
let enableWebnnDelegate = false;
const disabledSelectors = ['#tabs > li', '.btn'];

$(document).ready(async () => {
  await tf.setBackend('wasm');
  await tf.ready();
  $('.icdisplay').hide();
});

$('input[name="model"]').on('change', async (e) => {
  modelChanged = true;
  modelName = $(e.target).attr('id');
  if (modelName.startsWith('selfie')) {
    $('#deeplabModelBtns .btn').removeClass('active');
  } else {
    $('#ssModelsBtns .btn').removeClass('active');
  }
  inputOptions.inputDimensions = modelConfigs[modelName].inputDimensions;
  inputOptions.inputResolution = modelConfigs[modelName].inputResolution;
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  await main();
});

$('#webnnDelegate').on('change', async (e) => {
  modelChanged = true;
  enableWebnnDelegate = $(e.target)[0].checked;
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
  const inputBuffer = utils.getInputTensor(camElement, inputOptions);
  console.log('- Computing... ');
  const start = performance.now();
  const result =
      await postAndListenMessage({action: 'compute', buffer: inputBuffer});
  computeTime = (performance.now() - start).toFixed(2);
  outputBuffer = result.outputBuffer;
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(outputBuffer, inputCanvas);
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

async function drawOutput(outputBuffer, srcElement) {
  if (modelName.startsWith('deeplab')) {
    // Do additional `argMax` for DeepLabV3 model
    outputBuffer = tf.tidy(() => {
      const a = tf.tensor(outputBuffer, [1, 257, 257, 21], 'float32');
      const b = tf.argMax(a, 3);
      const c = tf.tensor(b.dataSync(), b.shape, 'float32');
      return c.dataSync();
    });
  }
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
  $('#loadTime').html(`${loadTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime} ms`);
  }
}

async function postAndListenMessage(postedMessage) {
  if (postedMessage.action == 'compute') {
    // Transfer buffer rather than copy
    worker.postMessage(postedMessage, [postedMessage.buffer.buffer]);
  } else {
    worker.postMessage(postedMessage);
  }

  const result = await new Promise((resolve) => {
    worker.onmessage = (event) => {
      resolve(event.data);
    };
  });
  return result;
}

export async function main() {
  try {
    if (modelName === '') return;
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    const numRuns = utils.getUrlParams()[0];
    // Only do load() when model first time loads and
    // there's new model or delegate choosed
    if (isFirstTimeLoad || modelChanged) {
      modelChanged = false;
      isFirstTimeLoad = false;
      console.log(`- Model: ${modelName}-`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading model... ');
      const options = {
        action: 'load',
        modelPath: modelConfigs[modelName].modelPath,
        enableWebNNDelegate: enableWebnnDelegate,
        webNNDevicePreference: 0,
      };
      loadTime = await postAndListenMessage(options);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;

      console.log('- Warmup... ');
      const result =
          await postAndListenMessage({action: 'compute', buffer: inputBuffer});
      console.log('- Warmup done... ');

      for (let i = 0; i < numRuns; i++) {
        const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
        const start = performance.now();
        await postAndListenMessage({action: 'compute', buffer: inputBuffer});
        const time = performance.now() - start;
        console.log(`  compute time ${i+1}: ${time.toFixed(2)} ms`);
        computeTimeArray.push(time);
      }
      computeTime = utils.getMedianValue(computeTimeArray);
      computeTime = computeTime.toFixed(2);
      if (numRuns > 1) {
        medianComputeTime = computeTime;
      }
      outputBuffer = result.outputBuffer;
      console.log('outputBuffer: ', outputBuffer);

      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(outputBuffer, imgElement);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      stream = await utils.getMediaStream();
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
