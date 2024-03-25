'use strict';

import * as utils from '../common/utils.js';
import {LeNet} from './lenet.js';
import {Pen} from './pen.js';
import {addAlert} from '../common/ui.js';

const buildTimeElement = document.getElementById('buildTime');
const inferenceTimeElement = document.getElementById('inferenceTime');
const predictButton = document.getElementById('predict');
const nextButton = document.getElementById('next');
const clearButton = document.getElementById('clear');
const visualCanvas = document.getElementById('visual_canvas');
const visualContext = visualCanvas.getContext('2d');
const digitCanvas = document.createElement('canvas');
digitCanvas.setAttribute('height', 28);
digitCanvas.setAttribute('width', 28);
digitCanvas.style.backgroundColor = 'black';
const digitContext = digitCanvas.getContext('2d');
const pen = new Pen(visualCanvas);
let lenet;
let numRuns;

function clearInferenceResult() {
  inferenceTimeElement.innerHTML = '';
  for (let i = 0; i < 3; ++i) {
    const labelElement = document.getElementById(`label${i}`);
    const probElement = document.getElementById(`prob${i}`);
    labelElement.innerHTML = '';
    probElement.innerHTML = '';
  }
}

$('#backendBtns .btn').on('change', async () => {
  await main();
});

function drawNextDigitFromMnist() {
  const n = Math.floor(Math.random() * 10);
  const digit = mnist[n].get();
  mnist.draw(digit, digitContext);
  visualContext.drawImage(
      digitCanvas, 0, 0, visualCanvas.width, visualCanvas.height);
}

function getInputFromCanvas() {
  digitContext.clearRect(0, 0, digitCanvas.width, digitCanvas.height);
  digitContext.drawImage(
      visualCanvas, 0, 0, digitCanvas.width, digitCanvas.height);
  const imageData =
      digitContext.getImageData(0, 0, digitCanvas.width, digitCanvas.height);
  const input = new Float32Array(digitCanvas.width * digitCanvas.height);
  for (let i = 0; i < input.length; i++) {
    input[i] = imageData.data[i * 4];
  }
  return input;
}

function getMedianValue(array) {
  array = array.sort((a, b) => a - b);
  return array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
      (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
}

async function main() {
  buildTimeElement.innerHTML = '';
  predictButton.setAttribute('disabled', true);
  clearInferenceResult();
  const [backend, deviceType] =
      $('input[name="backend"]:checked').attr('id').split('_');
  await utils.setBackend(backend, deviceType);
  drawNextDigitFromMnist();
  const weightUrl = utils.weightsOrigin() +
    '/test-data/models/lenet_nchw/weights/lenet.bin';
  const layout = deviceType === 'cpu' ? 'nhwc' : 'nchw';
  lenet = new LeNet(weightUrl, layout);
  const [localNumRuns, powerPreference, numThreads] = utils.getUrlParams();
  numRuns = localNumRuns;
  try {
    const contextOptions = {deviceType};
    if (powerPreference) {
      contextOptions['powerPreference'] = powerPreference;
    }
    if (numThreads) {
      contextOptions['numThreads'] = numThreads;
    }
    let start = performance.now();
    const outputOperand = await lenet.load(contextOptions);
    console.log(
        `loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    start = performance.now();
    await lenet.build(outputOperand);
    const buildTime = performance.now() - start;
    console.log(`build elapsed time: ${buildTime.toFixed(2)} ms`);
    buildTimeElement.innerHTML = 'Build Time: ' +
        `<span class='text-primary'>${buildTime.toFixed(2)}</span> ms`;

    predictButton.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addAlert(error.message);
  }
}

predictButton.addEventListener('click', async function(e) {
  clearInferenceResult();
  predictButton.setAttribute('disabled', true);
  try {
    let start;
    let inferenceTime;
    const inferenceTimeArray = [];
    const input = getInputFromCanvas();
    let outputBuffer = new Float32Array(utils.sizeOfShape([1, 10]));

    // Do warm up
    let results = await lenet.compute(input, outputBuffer);

    for (let i = 0; i < numRuns; i++) {
      start = performance.now();
      results = await lenet.compute(
          results.inputs.input, results.outputs.output);
      inferenceTime = performance.now() - start;
      console.log(`execution elapsed time: ${inferenceTime.toFixed(2)} ms`);
      inferenceTimeArray.push(inferenceTime);
    }

    if (numRuns === 1) {
      inferenceTimeElement.innerHTML = 'Execution Time: ' +
          `<span class='text-primary'>${inferenceTime.toFixed(2)}</span> ms`;
    } else {
      const medianInferenceTime = getMedianValue(inferenceTimeArray);
      console.log(`median execution elapsed time: ` +
          `${medianInferenceTime.toFixed(2)} ms`);
      inferenceTimeElement.innerHTML = `Median Execution Time(${numRuns}` +
          ` runs): <span class='text-primary'>` +
          `${medianInferenceTime.toFixed(2)}</span> ms`;
    }

    outputBuffer = results.outputs.output;
    const classes = topK(Array.from(outputBuffer));
    classes.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      const labelElement = document.getElementById(`label${i}`);
      const probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });

    predictButton.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addAlert(error.message);
  }
});

nextButton.addEventListener('click', () => {
  drawNextDigitFromMnist();
  clearInferenceResult();
});

clearButton.addEventListener('click', () => {
  pen.clear();
  clearInferenceResult();
});

function topK(probs, k = 3) {
  const sorted = probs.map((prob, index) => [prob, index]).sort((a, b) => {
    if (a[0] === b[0]) {
      return 0;
    }
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();

  const classes = [];
  for (let i = 0; i < k; ++i) {
    const c = {
      label: sorted[i][1],
      prob: (sorted[i][0] * 100).toFixed(2),
    };
    classes.push(c);
  }

  return classes;
}
