'use strict';

import {LeNet} from './lenet.js';
import {Pen} from './pen.js';

const compilationTimeElement = document.getElementById('compilationTime');
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

function getMedianValue(arr) {
  return arr.length % 2 !== 0 ? arr[Math.floor(arr.length / 2)] :
      (arr[arr.length / 2 - 1] + arr[arr.length / 2]) / 2;
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
  const lenet = new LeNet('lenet.bin');
  try {
    let start = performance.now();
    await lenet.load();
    console.log(
        `loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    start = performance.now();
    await lenet.compile();
    const compilationTime = performance.now() - start;
    console.log(`compilation elapsed time: ${compilationTime.toFixed(2)} ms`);
    compilationTimeElement.innerHTML = 'Compilation Time: ' +
        `<span class='text-primary'>${compilationTime.toFixed(2)}</span> ms`;

    predictButton.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addWarning(error.message);
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
      let result;
      let inferenceTime;
      const inferenceTimeArr = [];
      const input = getInputFromCanvas();

      for (let i = 0; i < n; i++) {
        start = performance.now();
        result = await lenet.predict(input);
        inferenceTime = performance.now() - start;
        console.log(`execution elapsed time: ${inferenceTime.toFixed(2)} ms`);
        console.log(`execution result: ${result}`);
        inferenceTimeArr.push(inferenceTime);
      }

      if (n === 1) {
        inferenceTimeElement.innerHTML = 'Execution Time: ' +
        `<span class='text-primary'>${inferenceTime.toFixed(2)}</span> ms`;
      } else {
        const medianInferenceTime = getMedianValue(inferenceTimeArr);
        console.log(`median execution elapsed time: ` +
            `${medianInferenceTime.toFixed(2)} ms`);
        inferenceTimeElement.innerHTML = `Median Execution Time(${n} runs): ` +
            `<span class='text-primary'>${medianInferenceTime.toFixed(2)}` +
            '</span> ms';
      }

      const classes = topK(Array.from(result));
      classes.forEach((c, i) => {
        console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
        const labelElement = document.getElementById(`label${i}`);
        const probElement = document.getElementById(`prob${i}`);
        labelElement.innerHTML = `${c.label}`;
        probElement.innerHTML = `${c.prob}%`;
      });
    } catch (error) {
      console.log(error);
      addWarning(error.message);
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

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}
