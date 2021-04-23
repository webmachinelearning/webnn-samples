import {NSNet2} from './nsnet2.js';
import * as featurelib from './featurelib.js';

export class Denoiser {
  constructor(batchSize, frames, sampleRate) {
    this.cfg = {
      'winlen': 0.02,
      'hopfrac': 0.5,
      'fs': sampleRate,
      'mingain': -80,
      'feattype': 'LogPow',
    };
    this.batchSize = batchSize;
    this.frames = frames;
    this.nsnet = new NSNet2();
    this.mingain = 10**(this.cfg['mingain']/20);
    this.logger = null;
  }

  log(message, sep = false, append = true) {
    if (this.logger) {
      this.logger.innerHTML = (append ? this.logger.innerHTML : '') + message +
          (sep ? '<br>' : '');
    }
  }

  async prepare() {
    return new Promise((resolve, reject) => {
      this.log(' - Loading weights... ');
      const start = performance.now();
      const weightsUrl = 'https://webmachinelearning.github.io/test-data/' +
          'models/nsnet2_nchw/weights/';
      this.nsnet.load(
          weightsUrl, this.batchSize, this.frames).then((outputOperand) => {
        const modelLoadTime = performance.now() - start;
        this.log(`done in <span class='text-primary'>` +
            `${modelLoadTime.toFixed(2)}</span> ms.`, true);
        this.log(' - Building... ');
        setTimeout(async () => {
          try {
            const start = performance.now();
            await this.nsnet.build(outputOperand);
            const modelBuildTime = performance.now() - start;
            this.log(`done in <span class='text-primary'>` +
                `${modelBuildTime.toFixed(2)}</span> ms.`, true);
            this.log(' - Warming up iSTFT... ');
          } catch (error) {
            reject(error);
          }
          setTimeout(async () => {
            try {
              // warm up the spec2sig
              const start = performance.now();
              const outSpec = tf.zeros([161, this.frames], 'complex64');
              const sigOut = featurelib.spec2sig(outSpec, this.cfg);
              const spec2SigWarmupTime = performance.now() - start;
              this.log(`done in <span class='text-primary'>` +
                  `${spec2SigWarmupTime.toFixed(2)}</span> ms.`, true);
              await sigOut.data();
              outSpec.dispose();
              sigOut.dispose();
              resolve();
            } catch (error) {
              reject(error);
            }
          }, 0);
        }, 0);
      }).catch((error) => {
        reject(error);
      });
    });
  }

  async process(audioData, callback) {
    const sizePerFrame = 160;
    const audioFrames = Math.floor(audioData.length / sizePerFrame);
    const audioTensor = tf.tensor1d(audioData);
    // Set the overlap for two adjacent frames that avoids sound breaks.
    const overlap = 1;
    const processStart = performance.now();
    let lastIteration = false;
    let initialHiddenState92Buffer =
        new Float32Array(1 * this.batchSize * this.nsnet.hiddenSize);
    let initialHiddenState155Buffer =
        new Float32Array(1 * this.batchSize * this.nsnet.hiddenSize);
    for (let frame = 0; !lastIteration; frame += this.frames - overlap * 2) {
      lastIteration = frame + this.frames + 1 > audioFrames;
      const audioSize = sizePerFrame * (this.frames + 1);
      let endPadding = 0;
      const sigIn = tf.tidy(() => {
        let sigIn = audioTensor.slice(
            [sizePerFrame * frame], [lastIteration ? -1: audioSize]);
        if (sigIn.shape[0] < audioSize) {
          endPadding = audioSize - sigIn.shape[0];
          sigIn = sigIn.pad([[0, endPadding]]);
        }
        return sigIn;
      });
      let start = performance.now();
      const inputSpec = featurelib.calcSpec(sigIn, this.cfg);
      sigIn.dispose();
      const inputFeature = tf.tidy(() => {
        return featurelib.calcFeat(inputSpec, this.cfg).expandDims(0);
      });
      const inputData = await inputFeature.data();
      inputFeature.dispose();
      const calcFeatTime = (performance.now() - start).toFixed(2);
      start = performance.now();
      const outputs = await this.nsnet.compute(
          inputData, initialHiddenState92Buffer, initialHiddenState155Buffer);
      const computeTime = (performance.now() - start).toFixed(2);
      initialHiddenState92Buffer = outputs.gru94.data;
      initialHiddenState155Buffer = outputs.gru157.data;
      start = performance.now();
      let sliceStart;
      let sliceSize;
      const sigOut = tf.tidy(() => {
        const out = tf.tensor(outputs.output.data, outputs.output.dimensions);
        let Gain = tf.transpose(out);
        Gain = tf.clipByValue(Gain, this.mingain, 1.0);
        // Workaround tf.js WebGL backend for complex data.
        const inputSpecTransposed = tf.complex(
            tf.real(inputSpec).transpose(),
            tf.imag(inputSpec).transpose());
        const outSpec = tf.mul(inputSpecTransposed, Gain.squeeze());
        const sigOut = featurelib.spec2sig(outSpec, this.cfg);
        if (frame === 0) {
          sliceStart = 0;
          if (lastIteration) {
            sliceSize = sigOut.shape[0] - endPadding;
          } else {
            sliceSize = sigOut.shape[0] - overlap * sizePerFrame - sizePerFrame;
          }
        } else if (lastIteration) {
          sliceStart = overlap * sizePerFrame;
          sliceSize = sigOut.shape[0] - endPadding - overlap * sizePerFrame;
        } else {
          sliceStart = overlap * sizePerFrame;
          sliceSize = sigOut.shape[0] - 2 * overlap * sizePerFrame -
              sizePerFrame;
        }
        return sigOut.slice([sliceStart], [sliceSize]);
      });
      inputSpec.dispose();
      const sigData = await sigOut.data();
      sigOut.dispose();
      const spec2SigTime = (performance.now() - start).toFixed(2);
      callback(sigData);
      const progress = (frame + sliceSize / sizePerFrame) / audioFrames;
      this.log(
          `Denoising...  ` +
          `(${lastIteration ? 100 : Math.ceil(progress * 100)}%)<br>` +
          ` - STFT compute time: <span class='text-primary'>` +
          `${calcFeatTime}</span> ms.<br>` +
          ` - NSNet2 compute time: <span class='text-primary'>` +
          `${computeTime}</span> ms.<br>` +
          ` - iSTFT compute time: <span class='text-primary'>` +
          `${spec2SigTime}</span> ms.`, true, false);
    }
    audioTensor.dispose();
    const processTime = (performance.now() - processStart).toFixed(2);
    this.log(`<b>Done.</b> Processed ${audioFrames} ` +
        `frames in <span class='text-primary'>${processTime}</span> ms.`, true);
  }
}
