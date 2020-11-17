'use strict';

import {Denoiser} from './denoiser.js';
import {AudioPlayer} from './audio_player.js';

const sampleRate = 16000;
const batchSize = 1;
const defaultFrames = 50;
let denoiser;
let audioData;
let denoisedAudioData = [];
const playDenoisedButton = document.getElementById('play-denoised');
const playOriginalButton = document.getElementById('play-original');
const denoisedAudioPlayer = new AudioPlayer(sampleRate, playDenoisedButton, 'the denoised audio');
const originalAudioPlayer = new AudioPlayer(sampleRate, playOriginalButton, 'the original audio');

export async function main() {
  // Handle frames parameter.
  const searchParams = new URLSearchParams(location.search);
  frames = parseInt(searchParams.get('frames'));
  if (!frames) {
    // default
    frames = defaultFrames;
  }
  denoiser = new Denoiser(batchSize, frames, sampleRate);
  denoiser.logger = document.getElementById('info');
  denoiser.logger.innerHTML = `Creating NSNet2 with batch_size = ${batchSize} and frames = ${frames}.<br>`;
  await denoiser.prepare();
  denoiser.logger = document.getElementById('denoise-info');
  fileInputLabel.innerHTML = 'NSNet2 is ready.<br>Choose an audio file for noise suppresion.';
  fileInput.removeAttribute('disabled');
}

const fileInputLabel = document.getElementById('file-input-label');
const fileInput = document.getElementById('file-input');
fileInput.addEventListener('input', (event) => {
  originalAudioPlayer.stop();
  denoisedAudioPlayer.stop();
  playOriginalButton.setAttribute('disabled', true);
  fileInput.setAttribute('disabled', true);
  const input = event.target;
  const reader = new FileReader();
  reader.onload = async function() {
    playDenoisedButton.setAttribute('disabled', true);
    const arrayBuffer = reader.result;
    const audioContext = new AudioContext({sampleRate: denoiser.cfg.fs});
    let start = performance.now();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);
    console.log(`decode time: ${performance.now() - start}`);
    audioData = decoded.getChannelData(0);
    playOriginalButton.removeAttribute('disabled');
    originalAudioPlayer.play(new Float32Array(audioData));
    denoisedAudioData = [];
    await denoiser.process(audioData, (data) => {
      denoisedAudioData = denoisedAudioData.concat(Array.from(data));
    });
    console.log('denoise is done.');
    playDenoisedButton.removeAttribute('disabled');
    fileInput.removeAttribute('disabled');
  };
  reader.readAsArrayBuffer(input.files[0]);
});

playOriginalButton.onclick = () => {
  if (playOriginalButton.state === 'Pause') {
    originalAudioPlayer.pause();
  } else if (playOriginalButton.state === 'Resume') {
    denoisedAudioPlayer.pause();
    originalAudioPlayer.resume();
  } else if (playOriginalButton.state === 'Play') {
    denoisedAudioPlayer.pause();
    originalAudioPlayer.play(new Float32Array(audioData));
  }
};

playDenoisedButton.onclick = () => {
  if (playDenoisedButton.state === 'Pause') {
    denoisedAudioPlayer.pause();
  } else if (playDenoisedButton.state === 'Resume') {
    originalAudioPlayer.pause();
    denoisedAudioPlayer.resume();
  } else if (playDenoisedButton.state === 'Play') {
    originalAudioPlayer.pause();
    denoisedAudioPlayer.play(new Float32Array(denoisedAudioData));
  }
};
