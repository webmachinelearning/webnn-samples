'use strict';

import {Denoiser} from './denoiser.js';

const sampleRate = 16000;
const batchSize = 1;
const minFrames = 30;
const defaultFrames = 100;
let denoiser;
let audioData;
let denoisedAudioData = [];

const originalAudio = document.getElementById('original-audio');
const denoisedAudio = document.getElementById('denoised-audio');

originalAudio.onplay = () => {
  denoisedAudio.pause();
};

denoisedAudio.onplay = () => {
  originalAudio.pause();
};

const recorderWorker = new Worker('./libs/recorderWorker.js');
recorderWorker.postMessage({
  command: 'init',
  config: {sampleRate, numChannels: 1},
});

recorderWorker.onmessage = function( e ) {
  const blob = e.data;
  denoisedAudio.src = URL.createObjectURL(blob);
};

export async function main() {
  try {
    // Handle frames parameter.
    const searchParams = new URLSearchParams(location.search);
    let frames = parseInt(searchParams.get('frames'));
    if (!frames) {
      frames = defaultFrames;
    } else if (frames < minFrames) {
      frames = minFrames;
    }
    denoiser = new Denoiser(batchSize, frames, sampleRate);
    denoiser.logger = document.getElementById('info');
    denoiser.logger.innerHTML = `Creating NSNet2 with input shape ` +
        `[batch_size (${batchSize}) x frames (${frames}) x 161].<br>`;
    await denoiser.prepare();
    denoiser.logger.innerHTML += 'NSNet2 is <b>ready</b>.';
    denoiser.logger = document.getElementById('denoise-info');
    fileInput.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}

const fileInput = document.getElementById('file-input');
fileInput.addEventListener('input', (event) => {
  originalAudio.pause();
  denoisedAudio.pause();
  originalAudio.src = '';
  denoisedAudio.src = '';
  denoiser.logger.innerHTML = '';
  const input = event.target;
  if (input.files.length == 0) {
    return;
  }
  try {
    fileInput.setAttribute('disabled', true);
    const bufferReader = new FileReader();
    bufferReader.onload = async function(e) {
      const arrayBuffer = e.target.result;
      const audioContext = new AudioContext({sampleRate});
      const start = performance.now();
      const decoded = await audioContext.decodeAudioData(arrayBuffer);
      console.log(`decode time: ${performance.now() - start}`);
      audioData = decoded.getChannelData(0);
      denoisedAudioData = [];
      await denoiser.process(audioData, (data) => {
        denoisedAudioData = denoisedAudioData.concat(Array.from(data));
      });
      fileInput.removeAttribute('disabled');
      // Send the denoised audio data for wav encoding.
      recorderWorker.postMessage({
        command: 'clear',
      });
      recorderWorker.postMessage({
        command: 'record',
        buffer: [new Float32Array(denoisedAudioData)],
      });
      recorderWorker.postMessage({
        command: 'exportWAV',
        type: 'audio/wav',
      });
    };
    bufferReader.readAsArrayBuffer(input.files[0]);
    const fileReader = new FileReader();
    fileReader.onload = function(e) {
      originalAudio.src = e.target.result;
    };
    fileReader.readAsDataURL(input.files[0]);
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
});

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}
