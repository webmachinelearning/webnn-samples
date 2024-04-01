import {Processer} from './processer.js';
import {RNNoise} from './rnnoise.js';
import * as utils from '../common/utils.js';
import {addAlert} from '../common/ui.js';

const batchSize = 1;
const frames = 100; // Frames is fixed at 100
const frameSize = 480;
const gainsSize = 22;
const weightsUrl = utils.weightsOrigin() +
  '/test-data/models/rnnoise/weights/';
const rnnoise = new RNNoise(weightsUrl, batchSize, frames);

$('#backendBtns .btn').on('change', async () => {
  await main();
});

const sampleAudios = [{
  name: 'voice1',
  url: './audio/voice1.wav',
}, {
  name: 'voice2',
  url: './audio/voice2.wav',
}, {
  name: 'voice3',
  url: './audio/voice3.wav',
}];

const audioName = document.getElementById('audio-name');
const modelInfo = document.getElementById('info');
const DenoiseInfo = document.getElementById('denoise-info');
const fileInput = document.getElementById('file-input');
const originalAudio = document.getElementById('original-audio');
const denoisedAudio = document.getElementById('denoised-audio');
const recorderWorker = new Worker('./utils/recorderWorker.js');

recorderWorker.postMessage({
  command: 'init',
  config: {sampleRate: 48000, numChannels: 1},
});

recorderWorker.onmessage = function(e) {
  const blob = e.data;
  denoisedAudio.src = URL.createObjectURL(blob);
};

const wasmScript = document.createElement('script');
wasmScript.type = 'text/javascript';
wasmScript.onload = function() {
  console.log('WASM script loaded!');
  Module.onRuntimeInitialized = function() {
    console.log('WASM Runtime Ready.');
    console.log('DSP library Loaded.');
  };
};
wasmScript.src = 'process/process.js';
document.getElementsByTagName('head')[0].appendChild(wasmScript);

function getUrlById(audioList, id) {
  for (const audio of Object.values(audioList).flat()) {
    if (id === audio.name) {
      return audio.url;
    }
  }
  return null;
}

async function log(infoElement, message, sep = false, append = true) {
  await new Promise((resolve) => {
    setTimeout(() => {
      infoElement.innerHTML = (append ? infoElement.innerHTML : '') + message +
        (sep ? '<br>' : '');
      resolve();
    }, 0);
  });
}

originalAudio.onplay = () => {
  denoisedAudio.pause();
};

denoisedAudio.onplay = () => {
  originalAudio.pause();
};

async function denoise() {
  const audioData = [];
  const audioContext = new AudioContext({sampleRate: 48000});
  const vadInitialHiddenStateBuffer = new Float32Array(
      rnnoise.vadGruNumDirections * batchSize *rnnoise.vadGruHiddenSize,
  ).fill(0);
  const noiseInitialHiddenStateBuffer = new Float32Array(
      rnnoise.noiseGruNumDirections * batchSize * rnnoise.noiseGruHiddenSize,
  ).fill(0);
  const denoiseInitialHiddenStateBuffer = new Float32Array(
      rnnoise.denoiseGruNumDirections * batchSize *
      rnnoise.denoiseGruHiddenSize,
  ).fill(0);
  const inputs = {
    'input': null,
    'vadGruInitialH': vadInitialHiddenStateBuffer,
    'noiseGruInitialH': noiseInitialHiddenStateBuffer,
    'denoiseGruInitialH': denoiseInitialHiddenStateBuffer,
  };
  const outputBuffer = new Float32Array(batchSize * frames * gainsSize);
  const vadGruYHBuffer = new Float32Array(
      rnnoise.vadGruNumDirections * batchSize * rnnoise.vadGruHiddenSize);
  const noiseGruYHBuffer = new Float32Array(
      rnnoise.noiseGruNumDirections * batchSize * rnnoise.noiseGruHiddenSize);
  const denoiseGruYHBuffer = new Float32Array(
      rnnoise.denoiseGruNumDirections * batchSize *
      rnnoise.denoiseGruHiddenSize);
  let outputs = {
    'denoiseOutput': outputBuffer,
    'vadGruYH': vadGruYHBuffer,
    'noiseGruYH': noiseGruYHBuffer,
    'denoiseGruYH': denoiseGruYHBuffer,
  };

  if (audioContext.state != 'running') {
    audioContext.resume().then(function() {
      console.log('audioContext resumed.');
    });
  }
  const analyser = new Processer(audioContext, originalAudio, frames);
  const pcm = await analyser.getAudioPCMData();
  const inputSize = frameSize * frames;
  const numInputs = Math.ceil(pcm.length / inputSize);
  const lastInputSize = pcm.length - inputSize * (numInputs - 1);

  const processStart = performance.now();
  for (let i = 0; i < numInputs; i++) {
    let inputPCM;
    if (i != (numInputs - 1)) {
      inputPCM = pcm.subarray(i * inputSize, (i + 1) * inputSize);
    } else {
      inputPCM = new Float32Array(inputSize).fill(0);
      for (let j = 0; j < lastInputSize; j++) {
        inputPCM[j] = pcm[i * inputSize + j];
      }
    }
    let start = performance.now();
    const features = analyser.preProcessing(inputPCM);
    const preProcessingTime = (performance.now() - start).toFixed(2);
    inputs.input = new Float32Array(features);
    start = performance.now();
    outputs = await rnnoise.compute(inputs, outputs);
    const executionTime = (performance.now() - start).toFixed(2);
    inputs.vadGruInitialH = outputs.vadGruYH.slice();
    inputs.noiseGruInitialH = outputs.noiseGruYH.slice();
    inputs.denoiseGruInitialH = outputs.denoiseGruYH.slice();

    start = performance.now();
    const output = analyser.postProcessing(outputs.denoiseOutput);
    const postProcessingTime = (performance.now() - start).toFixed(2);
    audioData.push(...output);

    await log(
        DenoiseInfo, `Denoising...  ` +
        `(${Math.ceil((i + 1) / numInputs * 100)}%)<br>` +
        ` - preProcessing time: <span class='text-primary'>` +
        `${preProcessingTime}</span> ms.<br>` +
        ` - RNNoise compute time: <span class='text-primary'>` +
        `${executionTime}</span> ms.<br>` +
        ` - postProcessing time: <span class='text-primary'>` +
        `${postProcessingTime}</span> ms.`, true, false,
    );
  }
  const processTime = (performance.now() - processStart).toFixed(2);
  log(DenoiseInfo, `<b>Done.</b> Processed ${numInputs * 100} ` +
    `frames in <span class='text-primary'>${processTime}</span> ms.`, true);

  // Send the denoised audio data for wav encoding.
  recorderWorker.postMessage({
    command: 'clear',
  });
  recorderWorker.postMessage({
    command: 'record',
    buffer: [new Float32Array(audioData)],
  });
  recorderWorker.postMessage({
    command: 'exportWAV',
    type: 'audio/wav',
  });
}

$('.dropdown-item').click(async (e) => {
  const audioId = $(e.target).attr('id');
  if (audioId == 'browse') {
    const evt = document.createEvent('MouseEvents');
    evt.initEvent('click', true, false);
    fileInput.dispatchEvent(evt);
  } else {
    const audioUrl = getUrlById(sampleAudios, audioId);
    log(audioName,
        audioUrl.substring(audioUrl.lastIndexOf('/') + 1), false, false);
    originalAudio.src = audioUrl;
    denoisedAudio.src = '';
    await denoise();
  }
});

fileInput.addEventListener('input', (event) => {
  log(audioName, event.target.files[0].name, false, false);
  const reader = new FileReader();
  reader.onload = async function(e) {
    originalAudio.src = e.target.result;
    denoisedAudio.src = '';
    await denoise();
  };
  reader.readAsDataURL(event.target.files[0]);
});

export async function main() {
  try {
    const [backend, deviceType] =
        $('input[name="backend"]:checked').attr('id').split('_');
    await utils.setBackend(backend, deviceType);
    modelInfo.innerHTML = '';
    await log(modelInfo, `Creating RNNoise with input shape ` +
      `[${batchSize} (batch_size) x 100 (frames) x 42].`, true);
    await log(modelInfo, '- Loading model...');
    const powerPreference = utils.getUrlParams()[1];
    const contextOptions = {deviceType};
    if (powerPreference) {
      contextOptions['powerPreference'] = powerPreference;
    }
    const numThreads = utils.getUrlParams()[2];
    if (numThreads) {
      contextOptions['numThreads'] = numThreads;
    }
    let start = performance.now();
    const outputOperand = await rnnoise.load(contextOptions);
    const loadingTime = (performance.now() - start).toFixed(2);
    console.log(`loading elapsed time: ${loadingTime} ms`);
    await log(modelInfo,
        `done in <span class='text-primary'>${loadingTime}</span> ms.`, true);
    await log(modelInfo, '- Building model...');
    start = performance.now();
    await rnnoise.build(outputOperand);
    const buildTime = (performance.now() - start).toFixed(2);
    console.log(`build elapsed time: ${buildTime} ms`);
    await log(modelInfo,
        `done in <span class='text-primary'>${buildTime}</span> ms.`, true);
    await log(modelInfo, 'RNNoise is <b>ready</b>.');
    $('#choose-audio').attr('disabled', false);
  } catch (error) {
    console.log(error);
    addAlert(error.message);
  }
}
