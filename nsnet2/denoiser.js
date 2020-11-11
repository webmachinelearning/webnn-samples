import {NSNet2} from './nsnet2.js';
import * as featurelib from './featurelib.js';

export class Denoiser {
  constructor(sampleRate = 16000, batchSize = 1, frames = 100) {
    this.cfg = {
      'winlen'   : 0.02,
      'hopfrac'  : 0.5,
      'fs'       : sampleRate,
      'mingain'  : -80,
      'feattype' : 'LogPow'
    };
    this.batchSize = batchSize;
    this.frames = frames;
    this.nsnet = new NSNet2('./weights/', this.batchSize, this.frames);
    this.mingain = 10**(this.cfg['mingain']/20);
  }

  async prepare() {
    let start = performance.now();
    await this.nsnet.load();
    const modelLoadTime = performance.now() - start;
    console.log(`nsnet2 load time: ${modelLoadTime.toFixed(2)} ms`);
    start = performance.now();
    await this.nsnet.compile();
    const modelCompileTime = performance.now() - start;
    console.log(`nsnet2 compile time: ${modelCompileTime.toFixed(2)} ms`);
    // warm up the spec2sig
    start = performance.now();
    const outSpec = tf.zeros([161, this.frames], 'complex64');
    const sigOut = featurelib.spec2sig(outSpec, this.cfg);
    const spec2SigWarmupTime = performance.now() - start;
    await sigOut.data();
    outSpec.dispose();
    sigOut.dispose();
    console.log(`spec2sig wramup time: ${spec2SigWarmupTime.toFixed(2)} ms`);
    return {modelLoadTime, modelCompileTime, spec2SigWarmupTime};
  }

  async process(audioData, callback) {
    const audioFrames = Math.floor(audioData.length / 160);
    const audioTensor = tf.tensor1d(audioData);
    const overlap = 5;
    let start;
    for (let frame = 0; frame < audioFrames; frame += this.frames - overlap * 2) {
      const lastFrame = frame + this.frames + 1 > audioFrames;
      const audioSize = 160 * (this.frames + 1);
      let sigIn = audioTensor.slice([160 * frame], [lastFrame ? -1: audioSize]);
      let endPadding = 0;
      if (sigIn.shape[0] < audioSize) {
        endPadding = audioSize - sigIn.shape[0];
        sigIn = sigIn.pad([[0, endPadding]]);
      }
      start = performance.now();
      const inputSpec = featurelib.calcSpec(sigIn, this.cfg);
      console.log(`inputSpec shape ${inputSpec.shape}`);
      console.log(`calcuate spec time: ${performance.now() - start}`);
      const inputFeature = tf.tidy(() => {
        // Workaround tf.js WebGL backend for complex data.
        start = performance.now();
        let feature = featurelib.calcFeat(inputSpec, this.cfg);
        console.log(`calcuate feature time: ${performance.now() - start}`);
        return feature.expandDims(0);
      });
      console.log(`inputFeature shape ${inputFeature.shape}`);
      const inputData = await inputFeature.data();
      start = performance.now();
      const output = await this.nsnet.compute(inputData);
      console.log(`nsnet2 compute time: ${performance.now() - start}`);
      let outSpec = tf.tidy(() => {
        let out = tf.tensor(output.buffer, output.dimensions);
        let Gain = tf.transpose(out);
        Gain = tf.clipByValue(Gain, this.mingain, 1.0);
        // Workaround tf.js WebGL backend for complex data.
        const inputSpecTransposed = tf.complex(
          tf.real(inputSpec).transpose(),
          tf.imag(inputSpec).transpose());
        return tf.mul(inputSpecTransposed, Gain.squeeze());
      });
      start = performance.now();
      let sigOut = featurelib.spec2sig(outSpec, this.cfg);
      if (frame === 0) {
        sigOut = sigOut.slice([0], [sigOut.shape[0] - overlap * 160 - 160]);
      } else if (lastFrame) {
        sigOut = sigOut.slice([overlap * 160], [sigOut.shape[0] - endPadding - overlap * 160]);
      } else {
        sigOut = sigOut.slice([overlap * 160], [sigOut.shape[0] - 2 * overlap * 160 - 160]);
      }
      const sigData = await sigOut.data();
      console.log(`spec to signal time: ${performance.now() - start}`);
      callback(sigData, this.frames, frame, audioFrames);
      outSpec.dispose();
      sigOut.dispose();
      inputSpec.dispose();
      inputFeature.dispose();
      sigIn.dispose();
    }
    audioTensor.dispose();
  }

  dispose() {
    this.nsnet.dispose();
  }
}