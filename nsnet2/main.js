'use strict';

import {NSNet2} from './nsnet2.js';
import * as featurelib from './featurelib.js';

class Denoiser {
  constructor() {
    this.cfg = {
      'winlen'   : 0.02,
      'hopfrac'  : 0.5,
      'fs'       : 16000,
      'mingain'  : -80,
      'feattype' : 'LogPow'
    };
    this.batchSize = 1;
    this.frames = 480;
    this.nsnet = new NSNet2('./weights/', this.batchSize, this.frames);
  }

  async prepare() {
    let start = performance.now();
    await this.nsnet.load();
    console.log(`nsnet2 load time: ${performance.now() - start}`);
    start = performance.now();
    await this.nsnet.compile();
    console.log(`nsnet2 compile time: ${performance.now()- start}`);
    start = performance.now();
    const outSpec = tf.zeros([161, 1], 'complex64');
    const sigOut = featurelib.spec2sig(outSpec, this.cfg);
    await sigOut.data();
    console.log(`spec to signal time: ${performance.now() - start}`);
  }

  async process(sigIn, callback) {
    let start = performance.now();
    const spec = featurelib.calcSpec(sigIn, this.cfg);
    console.log(`calcuate spec time: ${performance.now() - start}`);
    const specFrames = spec.shape[0];
    for (let frame = 0; frame < specFrames; frame += this.frames) {
      // Workaround tf.js WebGL backend for complex data.
      const size = frame + this.frames <= specFrames ? this.frames : specFrames - frame;
      const inputSpec = tf.complex(tf.real(spec).slice([frame], [size]),
                                   tf.imag(spec).slice([frame], [size]));
      start = performance.now();
      let inputFeature = featurelib.calcFeat(inputSpec, this.cfg);
      console.log(`calcuate feature time: ${performance.now() - start}`);
      if (size !== this.frames) {
        inputFeature = inputFeature.pad([[0, this.frames - size], [0, 0]])
      }
      inputFeature = inputFeature.expandDims(0);
      start = performance.now();
      const output = await this.nsnet.compute(await inputFeature.data());
      console.log(`nsnet2 compute time: ${performance.now() - start}`);
      let out = tf.tensor(output.buffer, output.dimensions);
      if (size !== this.frames) {
        out = out.slice([0, 0], [-1, size]);
      }
      let Gain = tf.transpose(out);
      Gain = tf.clipByValue(Gain, this.cfg.mingain, 1.0);
      // Workaround tf.js WebGL backend for complex data.
      const inputSpecTransposed = tf.complex(
        tf.real(spec).slice([frame], [size]).transpose(),
        tf.imag(spec).slice([frame], [size]).transpose());
      const outSpec = tf.mul(inputSpecTransposed, Gain.squeeze());
      start = performance.now();
      const sigOut = featurelib.spec2sig(outSpec, this.cfg);
      const sigData = await sigOut.data();
      console.log(`spec to signal time: ${performance.now() - start}`);
      callback(sigData);
    }
  }
}

function playF32Audio(f32buffer, inSampleRate) { 
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: inSampleRate});
  const myArrayBuffer = audioCtx.createBuffer(1, f32buffer.length, inSampleRate);
  
  myArrayBuffer.copyToChannel(f32buffer, 0, 0);

  const source = audioCtx.createBufferSource();
  source.buffer = myArrayBuffer;
  
  source.connect(audioCtx.destination);
  source.start();
}

let denoiser;

export async function main() {
  denoiser = new Denoiser();
  await denoiser.prepare();
  fileInput.removeAttribute('disabled');
}

let denoisedData = [];

const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('input', (event) => {
  const input = event.target;
  const reader = new FileReader();
  reader.onload = async function() {
    playButton.setAttribute('disabled', true);
    const arrayBuffer = reader.result;
    const audioContext = new AudioContext({sampleRate: denoiser.cfg.fs});
    let start = performance.now();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);
    console.log(`decode time: ${performance.now() - start}`);
    const sigIn = decoded.getChannelData(0);
    denoisedData = [];
    await denoiser.process(sigIn, (data) => {
      denoisedData = denoisedData.concat(Array.from(data));
    });
    console.log('denoise is done.');
    playButton.removeAttribute('disabled');
  };
  reader.readAsArrayBuffer(input.files[0]);
});

const playButton = document.getElementById('play');
playButton.addEventListener('click', ()=> {
  playF32Audio(new Float32Array(denoisedData), 16000);
});
