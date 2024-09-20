'use strict';

import {buildConstantByNpy} from '../common/utils.js';

export class RNNoise {
  constructor(modelPath, batchSize, frames) {
    this.baseUrl_ = modelPath;
    this.batchSize_ = batchSize;
    this.frames_ = frames;
    this.gainsSize_ = 22;
    this.model_ = null;
    this.context_ = null;
    this.graph_ = null;
    this.builder_ = null;
    this.inputTensor_ = null;
    this.vadGruInitialHTensor_ = null;
    this.noiseGruInitialHTensor_ = null;
    this.denoiseGruInitialHTensor_ = null;
    this.denoiseOutputTensor_ = null;
    this.vadGruYHTensor_ = null;
    this.noiseGruYHTensor_ = null;
    this.denoiseGruYHTensor_ = null;
    this.featureSize = 42;
    this.vadGruHiddenSize = 24;
    this.vadGruNumDirections = 1;
    this.noiseGruHiddenSize = 48;
    this.noiseGruNumDirections = 1;
    this.denoiseGruHiddenSize = 96;
    this.denoiseGruNumDirections = 1;
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    // Create constants by loading pre-trained data from .npy files.
    const inputDenseKernel0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'input_dense_kernel_0.npy');
    const inputDenseBias0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'input_dense_bias_0.npy');
    const vadGruW = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'vad_gru_W.npy');
    const vadGruR = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'vad_gru_R.npy');
    const vadGruBData = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'vad_gru_B.npy');
    const noiseGruW = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'noise_gru_W.npy');
    const noiseGruR = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'noise_gru_R.npy');
    const noiseGruBData = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'noise_gru_B.npy');
    const denoiseGruW = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_gru_W.npy');
    const denoiseGruR = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_gru_R.npy');
    const denoiseGruBData = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_gru_B.npy');
    const denoiseOutputKernel0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_output_kernel_0.npy');
    const denoiseOutputBias0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_output_bias_0.npy');
    // Build up the network.
    const inputDesc = {
      dataType: 'float32',
      dimensions: [this.batchSize_, this.frames_, this.featureSize],
      shape: [this.batchSize_, this.frames_, this.featureSize],
    };
    const input = this.builder_.input('input', inputDesc);
    inputDesc.usage = MLTensorUsage.WRITE;
    this.inputTensor_ = await this.context_.createTensor(inputDesc);

    const inputDense0 = this.builder_.matmul(input, inputDenseKernel0);
    const biasedTensorName2 = this.builder_.add(inputDense0, inputDenseBias0);
    const inputDenseTanh0 = this.builder_.tanh(biasedTensorName2);
    const vadGruX = this.builder_.transpose(
        inputDenseTanh0, {permutation: [1, 0, 2]});
    const vadGruB = this.builder_.slice(
        vadGruBData, [0, 0], [1, 3 * this.vadGruHiddenSize]);
    const vadGruRB = this.builder_.slice(
        vadGruBData,
        [0, 3 * this.vadGruHiddenSize],
        [1, 3 * this.vadGruHiddenSize]);

    const vadGruInitialHDesc = {
      dataType: 'float32',
      dimensions: [1, this.batchSize_, this.vadGruHiddenSize],
      shape: [1, this.batchSize_, this.vadGruHiddenSize],
    };
    const vadGruInitialH = this.builder_.input(
        'vadGruInitialH', vadGruInitialHDesc);
    vadGruInitialHDesc.usage = MLTensorUsage.WRITE;
    this.vadGruInitialHTensor_ = await this.context_.createTensor(
        vadGruInitialHDesc);

    const [vadGruYH, vadGruY] = this.builder_.gru(vadGruX,
        vadGruW, vadGruR, this.frames_, this.vadGruHiddenSize, {
          bias: vadGruB,
          recurrentBias: vadGruRB,
          initialHiddenState: vadGruInitialH,
          returnSequence: true,
          resetAfter: false,
          activations: ['sigmoid', 'relu'],
        });
    const vadGruYTransposed = this.builder_.transpose(
        vadGruY, {permutation: [2, 0, 1, 3]});
    const vadGruTranspose1 = this.builder_.reshape(
        vadGruYTransposed, [1, this.frames_, this.vadGruHiddenSize]);
    const concatenate1 = this.builder_.concat(
        [inputDenseTanh0, vadGruTranspose1, input], 2);
    const noiseGruX = this.builder_.transpose(
        concatenate1, {permutation: [1, 0, 2]});
    const noiseGruB = this.builder_.slice(
        noiseGruBData, [0, 0], [1, 3 * this.noiseGruHiddenSize]);
    const noiseGruRB = this.builder_.slice(
        noiseGruBData,
        [0, 3 * this.noiseGruHiddenSize],
        [1, 3 * this.noiseGruHiddenSize]);

    const noiseGruInitialHDesc = {
      dataType: 'float32',
      dimensions: [1, this.batchSize_, this.noiseGruHiddenSize],
      shape: [1, this.batchSize_, this.noiseGruHiddenSize],
    };
    const noiseGruInitialH = this.builder_.input(
        'noiseGruInitialH', noiseGruInitialHDesc);
    noiseGruInitialHDesc.usage = MLTensorUsage.WRITE;
    this.noiseGruInitialHTensor_ = await this.context_.createTensor(
        noiseGruInitialHDesc);

    const [noiseGruYH, noiseGruY] = this.builder_.gru(noiseGruX,
        noiseGruW, noiseGruR, this.frames_, this.noiseGruHiddenSize, {
          bias: noiseGruB,
          recurrentBias: noiseGruRB,
          initialHiddenState: noiseGruInitialH,
          returnSequence: true,
          resetAfter: false,
          activations: ['sigmoid', 'relu'],
        });
    const noiseGruYTransposed = this.builder_.transpose(
        noiseGruY, {permutation: [2, 0, 1, 3]});
    const noiseGruTranspose1 = this.builder_.reshape(
        noiseGruYTransposed, [1, this.frames_, this.noiseGruHiddenSize]);
    const concatenate2 = this.builder_.concat(
        [vadGruTranspose1, noiseGruTranspose1, input], 2);
    const denoiseGruX = this.builder_.transpose(
        concatenate2, {permutation: [1, 0, 2]});
    const denoiseGruB = this.builder_.slice(
        denoiseGruBData, [0, 0], [1, 3 * this.denoiseGruHiddenSize]);
    const denoiseGruRB = this.builder_.slice(
        denoiseGruBData,
        [0, 3 * this.denoiseGruHiddenSize],
        [1, 3 * this.denoiseGruHiddenSize]);

    const denoiseGruInitialHDesc = {
      dataType: 'float32',
      dimensions: [1, this.batchSize_, this.denoiseGruHiddenSize],
      shape: [1, this.batchSize_, this.denoiseGruHiddenSize],
    };
    const denoiseGruInitialH = this.builder_.input(
        'denoiseGruInitialH', denoiseGruInitialHDesc);
    denoiseGruInitialHDesc.usage = MLTensorUsage.WRITE;
    this.denoiseGruInitialHTensor_ = await this.context_.createTensor(
        denoiseGruInitialHDesc);

    const [denoiseGruYH, denoiseGruY] = this.builder_.gru(denoiseGruX,
        denoiseGruW, denoiseGruR, this.frames_, this.denoiseGruHiddenSize, {
          bias: denoiseGruB,
          recurrentBias: denoiseGruRB,
          initialHiddenState: denoiseGruInitialH,
          returnSequence: true,
          resetAfter: false,
          activations: ['sigmoid', 'relu'],
        });
    const denoiseGruYTransposed = this.builder_.transpose(
        denoiseGruY, {permutation: [2, 0, 1, 3]});
    const denoiseGruTranspose1 = this.builder_.reshape(
        denoiseGruYTransposed, [1, this.frames_, this.denoiseGruHiddenSize]);
    const denoiseOutput0 = this.builder_.matmul(
        denoiseGruTranspose1, denoiseOutputKernel0);
    const biasedTensorName = this.builder_.add(
        denoiseOutput0, denoiseOutputBias0);
    const denoiseOutput = this.builder_.sigmoid(biasedTensorName);

    const denoiseOutputShape =
        [this.batchSize_, this.frames_, this.gainsSize_];
    this.denoiseOutputTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: denoiseOutputShape,
      shape: denoiseOutputShape,
      usage: MLTensorUsage.READ,
    });
    const vadGruYHOutputShape =
        [this.vadGruNumDirections, this.batchSize_, this.vadGruHiddenSize];
    this.vadGruYHTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: vadGruYHOutputShape,
      shape: vadGruYHOutputShape,
      usage: MLTensorUsage.READ,
    });
    const noiseGruYHOutputShape =
        [this.noiseGruNumDirections, this.batchSize_, this.noiseGruHiddenSize];
    this.noiseGruYHTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: noiseGruYHOutputShape,
      shape: noiseGruYHOutputShape,
      usage: MLTensorUsage.READ,
    });
    const denoiseGruYHOutputShape = [
      this.denoiseGruNumDirections,
      this.batchSize_,
      this.denoiseGruHiddenSize,
    ];
    this.denoiseGruYHTensor_ = await this.context_.createTensor({
      dataType: 'float32',
      dimensions: denoiseGruYHOutputShape,
      shape: denoiseGruYHOutputShape,
      usage: MLTensorUsage.READ,
    });

    return {denoiseOutput, vadGruYH, noiseGruYH, denoiseGruYH};
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build(outputOperand);
  }

  async compute(inputs) {
    this.context_.writeTensor(this.inputTensor_, inputs.input);
    this.context_.writeTensor(
        this.vadGruInitialHTensor_, inputs.vadGruInitialH);
    this.context_.writeTensor(
        this.noiseGruInitialHTensor_, inputs.noiseGruInitialH);
    this.context_.writeTensor(
        this.denoiseGruInitialHTensor_, inputs.denoiseGruInitialH);
    const inputTensors = {
      'input': this.inputTensor_,
      'vadGruInitialH': this.vadGruInitialHTensor_,
      'noiseGruInitialH': this.noiseGruInitialHTensor_,
      'denoiseGruInitialH': this.denoiseGruInitialHTensor_,
    };
    const outputTensors = {
      'denoiseOutput': this.denoiseOutputTensor_,
      'vadGruYH': this.vadGruYHTensor_,
      'noiseGruYH': this.noiseGruYHTensor_,
      'denoiseGruYH': this.denoiseGruYHTensor_,
    };
    this.context_.dispatch(this.graph_, inputTensors, outputTensors);
    const results = {
      'denoiseOutput': new Float32Array(
          await this.context_.readTensor(this.denoiseOutputTensor_)),
      'vadGruYH': new Float32Array(
          await this.context_.readTensor(this.vadGruYHTensor_)),
      'noiseGruYH': new Float32Array(
          await this.context_.readTensor(this.noiseGruYHTensor_)),
      'denoiseGruYH': new Float32Array(
          await this.context_.readTensor(this.denoiseGruYHTensor_)),
    };
    return results;
  }
}
