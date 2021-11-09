'use strict';

import {Denoiser} from './denoiser.js';
import {setPolyfillBackend} from '../common/utils.js';
import {addAlert} from '../common/ui.js';

const sampleRate = 16000;
const batchSize = 1;
const minFrames = 5;
const defaultFrames = 10;
let denoiser;
let audioData;
let denoisedAudioData = [];
let devicePreference = 'gpu';

const chooseAudio = document.getElementById('choose-audio');
const audioName = document.getElementById('audio-name');

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');
  await setPolyfillBackend(devicePreference);
  await main();
});

const sampleAudios = [
  {
    name: 'babbel',
    url: './samples/babble_15dB.opus',
  },
  {
    name: 'car',
    url: './samples/car_15dB.opus',
  },
  {
    name: 'street',
    url: './samples/street_15dB.opus',
  },
];

for (const audio of sampleAudios) {
  const button = document.getElementById(audio.name);
  button.onclick = async () => {
    onNewFile();
    fileInput.value = '';
    audioName.innerHTML = audio.url.substring(audio.url.lastIndexOf('/') + 1);
    chooseAudio.setAttribute('disabled', true);
    originalAudio.src = audio.url;
    const response = await fetch(audio.url);
    const arrayBuffer = await response.arrayBuffer();
    denoise(arrayBuffer);
  };
}

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
  chooseAudio.removeAttribute('disabled');
};

const fileInput = document.getElementById('file-input');
fileInput.addEventListener('input', (event) => {
  onNewFile();
  const input = event.target;
  if (input.files.length == 0) {
    return;
  }
  audioName.innerHTML = input.files[0].name;
  try {
    chooseAudio.setAttribute('disabled', true);
    const bufferReader = new FileReader();
    bufferReader.onload = async function(e) {
      const arrayBuffer = e.target.result;
      denoise(arrayBuffer);
    };
    bufferReader.readAsArrayBuffer(input.files[0]);
    const fileReader = new FileReader();
    fileReader.onload = function(e) {
      originalAudio.src = e.target.result;
    };
    fileReader.readAsDataURL(input.files[0]);
  } catch (error) {
    console.log(error);
    addAlert(error.message);
  }
});

function onNewFile() {
  originalAudio.pause();
  denoisedAudio.pause();
  originalAudio.src = '';
  denoisedAudio.src = '';
  denoiser.logger.innerHTML = '';
  audioName.innerHTML = '';
}

const AudioContext = window.AudioContext || window.webkitAudioContext || false;

async function denoise(arrayBuffer) {
  const audioContext = new AudioContext({sampleRate});
  const start = performance.now();
  audioContext.decodeAudioData(arrayBuffer, async (decoded) => {
    console.log(`decode time: ${performance.now() - start}`);
    audioData = decoded.getChannelData(0);
    denoisedAudioData = [];
    await denoiser.process(audioData, (data) => {
      denoisedAudioData = denoisedAudioData.concat(Array.from(data));
    });

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
  });
}

const browseButton = document.getElementById('browse');
browseButton.onclick = () => {
  const evt = document.createEvent('MouseEvents');
  evt.initEvent('click', true, false);
  fileInput.dispatchEvent(evt);
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
        `[${batchSize} (batch_size) x ${frames} (frames) x 161].<br>`;
    await denoiser.prepare(devicePreference);
    denoiser.logger.innerHTML += 'NSNet2 is <b>ready</b>.';
    denoiser.logger = document.getElementById('denoise-info');
    chooseAudio.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addAlert(error.message);
  }
}
