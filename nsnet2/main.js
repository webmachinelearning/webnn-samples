'use strict';

import {Denoiser} from './denoiser.js';
import {AudioPlayer} from './audio_player.js';

const sampleRate = 16000;
let denoiser;
let audioData;
let denoisedAudioData = [];
const playDenoisedButton = document.getElementById('play-denoised');
const playOriginalButton = document.getElementById('play-original');
const denoisedAudioPlayer = new AudioPlayer(sampleRate, playDenoisedButton, 'the denoised audio');
const originalAudioPlayer = new AudioPlayer(sampleRate, playOriginalButton, 'the original audio');

export async function main() {
  denoiser = new Denoiser(sampleRate);
  const info = await denoiser.prepare();
  const infoElement = document.getElementById('info');
  infoElement.innerHTML = `NSNet2 configuration: batch_size=<span class='text-primary'>${denoiser.batchSize}</span>, ` + 
        `frames=<span class='text-primary'>${denoiser.frames}</span> <br>` +
        `Model load time: <span class='text-primary'>${info.modelLoadTime.toFixed(2)}</span> ms, ` +
        `model compile time: <span class='text-primary'>${info.modelCompileTime.toFixed(2)}</span> ms, ` +
        `spec2sig warmup time: <span class='text-primary'>${info.spec2SigWarmupTime.toFixed(2)}</span> ms.`;
  fileInput.removeAttribute('disabled');
}

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
    setTimeout(async () => {
      const denoiseInfo = document.getElementById('denoise-info');
      denoiseInfo.innerHTML = 'Processing...'
      await denoiser.process(audioData, (data, size, start, frames) => {
        console.log(`processed ${size} ${start}/${frames}`);
        denoisedAudioData = denoisedAudioData.concat(Array.from(data));
      });
      console.log('denoise is done.');
      playDenoisedButton.removeAttribute('disabled');
      fileInput.removeAttribute('disabled');
    }, 0);
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
