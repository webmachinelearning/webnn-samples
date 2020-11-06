'use strict';

import {NSNet} from './nsnet.js';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

function almostEqual(
    a, b, episilon = 1e-5, rtol = 5.0 * 1.1920928955078125e-7) {
  const delta = Math.abs(a - b);
  if (delta <= episilon + rtol * Math.abs(b)) {
    return true;
  } else {
    console.error(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

function checkValue(output, expected) {
  if (output.length !== expected.length) {
    console.error(`output length ${output.length} is not equal to ${expected.length}`);
  }
  for (let i = 0; i < output.length; ++i) {
    almostEqual(output[i], expected[i]);
  }
}

export async function main() {
  const batchSize = 1;
  const frames = 2077;
  const nsnet = new NSNet('./weights/', batchSize, frames);
  const inputData = await nsnet.fetchData('input.bin');
  const outData = await nsnet.fetchData('out.bin');
  let start = performance.now();
  await nsnet.load();
  console.log(`load time: ${performance.now() - start}`);
  start = performance.now();
  await nsnet.compile();
  console.log(`compile time: ${performance.now()- start}`);
  start = performance.now();
  const output = await nsnet.compute(inputData);
  console.log(`compute time: ${performance.now() - start}`);
  checkValue(output.buffer, outData);
  nsnet.dispose();
}

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}
