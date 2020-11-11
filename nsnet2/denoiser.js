import {NSNet2} from './nsnet2.js';
import * as featurelib from './featurelib.js';

export class Denoiser {
  constructor(sampleRate = 16000, batchSize = 1, frames = 10) {
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

  async process(sigIn, callback) {
    let start = performance.now();
    const spec = featurelib.calcSpec(sigIn, this.cfg);
    console.log(`calcuate spec time: ${performance.now() - start}`);
    const specFrames = spec.shape[0];
    for (let frame = 0; frame < specFrames; frame += this.frames) {
      // Workaround tf.js WebGL backend for complex data.
      const size = frame + this.frames <= specFrames ? this.frames : specFrames - frame;
      const inputFeature = tf.tidy(() => {
        const inputSpec = tf.complex(tf.real(spec).slice([frame], [size]),
                                    tf.imag(spec).slice([frame], [size]));
        start = performance.now();
        let inputFeature = featurelib.calcFeat(inputSpec, this.cfg);
        inputSpec.dispose();
        console.log(`calcuate feature time: ${performance.now() - start}`);
        if (size !== this.frames) {
          inputFeature = inputFeature.pad([[0, this.frames - size], [0, 0]])
        }
        return inputFeature.expandDims(0);
      });
      const inputData = await inputFeature.data();
      inputFeature.dispose();
      start = performance.now();
      const output = await this.nsnet.compute(inputData);
      console.log(`nsnet2 compute time: ${performance.now() - start}`);
      const outSpec = tf.tidy(() => {
        let out = tf.tensor(output.buffer, output.dimensions);
        if (size !== this.frames) {
          out = out.slice([0, 0], [-1, size]);
        }
        let Gain = tf.transpose(out);
        Gain = tf.clipByValue(Gain, this.mingain, 1.0);
        // Workaround tf.js WebGL backend for complex data.
        const inputSpecTransposed = tf.complex(
          tf.real(spec).slice([frame], [size]).transpose(),
          tf.imag(spec).slice([frame], [size]).transpose());
        return tf.mul(inputSpecTransposed, Gain.squeeze());
      });
      start = performance.now();
      const sigOut = featurelib.spec2sig(outSpec, this.cfg);
      const sigData = await sigOut.data();
      outSpec.dispose();
      sigOut.dispose();
      console.log(`spec to signal time: ${performance.now() - start}`);
      callback(sigData, this.frames, frame, specFrames);
    }
    spec.dispose();
  }

  dispose() {
    this.nsnet.dispose();
  }
}