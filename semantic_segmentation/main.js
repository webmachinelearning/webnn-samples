'use strict';

import {DeepLabV3MNV2Nchw} from './deeplabv3_mnv2_nchw.js';
import {DeepLabV3MNV2Nhwc} from './deeplabv3_mnv2_nhwc.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';
import {Renderer} from './lib/renderer.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
const outputCanvas = document.getElementById('outputCanvas');
let modelName ='deeplabv3mnv2';
let layout = 'nchw';
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
let outputBuffer;
let renderer;
let hoverPos = null;
let devicePreference = 'gpu';
let lastDevicePreference = '';
const disabledSelectors = ['#tabs > li', '.btn'];

$(document).ready(() => {
  $('.icdisplay').hide();
});

$(window).on('load', () => {
  renderer = new Renderer(outputCanvas);
  renderer.setup();
  loadRenderUI();
});

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

$('#modelBtns .btn').on('change', async (e) => {
  modelName = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

$('#layoutBtns .btn').on('change', async (e) => {
  layout = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  if (stream !== null) stopCamera();
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
  if (!isFirstTimeLoad) {
    await main();
  }
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('#pickimage').hide();
  $('.shoulddisplay').hide();
  await main();
});

function loadRenderUI() {
  const blurSlider = document.getElementById('blurSlider');
  const refineEdgeSlider = document.getElementById('refineEdgeSlider');
  const colorMapAlphaSlider = document.getElementById('colorMapAlphaSlider');
  const selectBackgroundButton = document.getElementById('chooseBackground');
  const clearBackgroundButton = document.getElementById('clearBackground');
  const colorPicker = new iro.ColorPicker('#color-picker-container', {
    width: 200,
    height: 200,
    color: {
      r: renderer.bgColor[0],
      g: renderer.bgColor[1],
      b: renderer.bgColor[2],
    },
    markerRadius: 5,
    sliderMargin: 12,
    sliderHeight: 20,
  });

  $('.bg-value').html(colorPicker.color.hexString);

  colorPicker.on('color:change', (color) => {
    $('.bg-value').html(color.hexString);
    renderer.bgColor = [color.rgb.r, color.rgb.g, color.rgb.b];
  });

  colorMapAlphaSlider.value = renderer.colorMapAlpha * 100;
  $('.color-map-alpha-value').html(renderer.colorMapAlpha);

  colorMapAlphaSlider.oninput = () => {
    const alpha = colorMapAlphaSlider.value / 100;
    $('.color-map-alpha-value').html(alpha);
    renderer.colorMapAlpha = alpha;
  };

  blurSlider.value = renderer.blurRadius;
  $('.blur-radius-value').html(renderer.blurRadius + 'px');

  blurSlider.oninput = () => {
    const blurRadius = parseInt(blurSlider.value);
    $('.blur-radius-value').html(blurRadius + 'px');
    renderer.blurRadius = blurRadius;
  };

  refineEdgeSlider.value = renderer.refineEdgeRadius;

  if (refineEdgeSlider.value === '0') {
    $('.refine-edge-value').html('DISABLED');
  } else {
    $('.refine-edge-value').html(refineEdgeSlider.value + 'px');
  }

  refineEdgeSlider.oninput = () => {
    const refineEdgeRadius = parseInt(refineEdgeSlider.value);
    if (refineEdgeRadius === 0) {
      $('.refine-edge-value').html('DISABLED');
    } else {
      $('.refine-edge-value').html(refineEdgeRadius + 'px');
    }
    renderer.refineEdgeRadius = refineEdgeRadius;
  };

  $('.effects-select .btn input').filter((e) => {
    return e.value === renderer.effect;
  }).parent().toggleClass('active');

  $('.controls').attr('data-select', renderer.effect);

  $('.effects-select .btn').click((e) => {
    e.preventDefault();
    const effect = e.target.children[0].value;
    $('.controls').attr('data-select', effect);
    renderer.effect = effect;
  });

  selectBackgroundButton.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      const img = new Image();
      img.onload = () => {
        renderer.backgroundImageSource = img;
      };
      img.src = URL.createObjectURL(files[0]);
    }
  }, false);

  clearBackgroundButton.addEventListener('click', (e) => {
    renderer.backgroundImageSource = null;
  }, false);

  outputCanvas.addEventListener('mousemove', (e) => {
    const getMousePos = (canvas, evt) => {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.ceil(evt.clientX - rect.left),
        y: Math.ceil(evt.clientY - rect.top),
      };
    };

    hoverPos = getMousePos(outputCanvas, e);
    renderer.highlightHoverLabel(hoverPos, outputCanvas);
  });

  outputCanvas.addEventListener('mouseleave', (e) => {
    hoverPos = null;
    renderer.highlightHoverLabel(hoverPos, outputCanvas);
  });
}

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

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
  const inputBuffer = utils.getInputTensor(camElement, inputOptions);
  const inputCanvas = utils.getVideoFrame(camElement);
  console.log('- Computing... ');
  const start = performance.now();
  await netInstance.compute(inputBuffer, outputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(inputCanvas);
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

async function drawOutput(srcElement) {
  // TODO: move 'argMax' operation to graph once it is supported in WebNN spec.
  // https://github.com/webmachinelearning/webnn/issues/184
  const argMaxResult = tf.tidy(() => {
    const a = tf.tensor(outputBuffer, netInstance.outputDimensions, 'float32');
    let axis = 3;
    if (layout === 'nchw') {
      axis = 1;
    }
    return tf.argMax(a, axis);
  });

  const width = inputOptions.inputDimensions[2];
  const imWidth = srcElement.naturalWidth | srcElement.width;
  const imHeight = srcElement.naturalHeight | srcElement.height;
  const resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
  const scaledWidth = Math.floor(imWidth / resizeRatio);
  const scaledHeight = Math.floor(imHeight / resizeRatio);

  const segMap = {
    data: await argMaxResult.data(),
    outputShape: argMaxResult.shape,
    labels: labels,
  };

  renderer.uploadNewTexture(srcElement, [scaledWidth, scaledHeight]);
  renderer.drawOutputs(segMap);
  renderer.highlightHoverLabel(hoverPos, outputCanvas);
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
    'deeplabv3mnv2nchw': new DeepLabV3MNV2Nchw(),
    'deeplabv3mnv2nhwc': new DeepLabV3MNV2Nhwc(),
  };

  return netObject[type];
}

export async function main() {
  try {
    ui.handleClick(disabledSelectors, true);
    let start;
    const [numRuns, powerPreference] = utils.getUrlParams();

    // Only do load() and build() when model first time loads,
    // there's new model choosed, and device backend changed
    if (isFirstTimeLoad || instanceType !== modelName + layout ||
      lastDevicePreference != devicePreference) {
      if (lastDevicePreference != devicePreference) {
        // Set polyfill backend
        await utils.setPolyfillBackend(devicePreference);
        lastDevicePreference = devicePreference;
      }
      if (netInstance !== null) {
        // Call dispose() to and avoid memory leak
        netInstance.dispose();
      }
      instanceType = modelName + layout;
      netInstance = constructNetObject(instanceType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);
      outputBuffer =
          new Float32Array(utils.sizeOfShape(netInstance.outputDimensions));
      isFirstTimeLoad = false;
      console.log(`- Model name: ${modelName}, Model layout: ${layout} -`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      const contextOptions = {devicePreference};
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
      if (numRuns > 1) {
        // Do warm up
        await netInstance.compute(inputBuffer, outputBuffer);
      }
      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        await netInstance.compute(inputBuffer, outputBuffer);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log('output: ', outputBuffer);
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(imgElement);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
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
