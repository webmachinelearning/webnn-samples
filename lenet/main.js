'use strict';

import {sizeOfShape, setPolyfillBackend} from '../common/utils.js';
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
let devicePreference = 'gpu';

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');
  await setPolyfillBackend(devicePreference);
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

function clearResult() {
  for (let i = 0; i < 3; ++i) {
    const labelElement = document.getElementById(`label${i}`);
    const probElement = document.getElementById(`prob${i}`);
    labelElement.innerHTML = '';
    probElement.innerHTML = '';
  }
}

export async function main() {
  drawNextDigitFromMnist();
  const pen = new Pen(visualCanvas);
  const weightUrl = '../test-data/models/lenet_nchw/weights/lenet.bin';
  const lenet = new LeNet(weightUrl);
  try {
    let start = performance.now();
    const outputOperand = await lenet.load(devicePreference);
    console.log(
        `loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    start = performance.now();
    lenet.build(outputOperand);
    const buildTime = performance.now() - start;
    console.log(`build elapsed time: ${buildTime.toFixed(2)} ms`);
    buildTimeElement.innerHTML = 'Build Time: ' +
        `<span class='text-primary'>${buildTime.toFixed(2)}</span> ms`;

    predictButton.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addAlert(error.message);
  }
  predictButton.addEventListener('click', async function(e) {
    try {
      const params = new URLSearchParams(location.search);
      const numRuns = params.get('numRuns');
      const n = numRuns === null ? 1 : parseInt(numRuns);

      if (n < 1) {
        alert(`The value of param numRuns must be greater than or equal to 1.`);
        return;
      }

      let start;
      let inferenceTime;
      const inferenceTimeArray = [];
      const input = getInputFromCanvas();
      const outputBuffer = new Float32Array(sizeOfShape([1, 10]));

      if (numRuns > 1) {
        // Do warm up
        lenet.predict(input, outputBuffer);
      }
      for (let i = 0; i < n; i++) {
        start = performance.now();
        lenet.predict(input, outputBuffer);
        inferenceTime = performance.now() - start;
        console.log(`execution elapsed time: ${inferenceTime.toFixed(2)} ms`);
        console.log(`execution result: ${outputBuffer}`);
        inferenceTimeArray.push(inferenceTime);
      }

      if (n === 1) {
        inferenceTimeElement.innerHTML = 'Execution Time: ' +
        `<span class='text-primary'>${inferenceTime.toFixed(2)}</span> ms`;
      } else {
        const medianInferenceTime = getMedianValue(inferenceTimeArray);
        console.log(`median execution elapsed time: ` +
            `${medianInferenceTime.toFixed(2)} ms`);
        inferenceTimeElement.innerHTML = `Median Execution Time(${n} runs): ` +
            `<span class='text-primary'>${medianInferenceTime.toFixed(2)}` +
            '</span> ms';
      }

      const classes = topK(Array.from(outputBuffer));
      classes.forEach((c, i) => {
        console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
        const labelElement = document.getElementById(`label${i}`);
        const probElement = document.getElementById(`prob${i}`);
        labelElement.innerHTML = `${c.label}`;
        probElement.innerHTML = `${c.prob}%`;
      });
    } catch (error) {
      console.log(error);
      addAlert(error.message);
    }
  });
  nextButton.addEventListener('click', () => {
    drawNextDigitFromMnist();
    clearResult();
  });

  clearButton.addEventListener('click', () => {
    pen.clear();
    clearResult();
  });
}

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
