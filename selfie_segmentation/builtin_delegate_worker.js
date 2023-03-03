'use strict';

/* eslint max-len: ["error", {"code": 120}] */

// built-in webnn delegate
importScripts('./tflite-support/tflite_model_runner_cc_simd.js');

let modelRunnerResult;
let modelRunner;
// Receive the message from the main thread
onmessage = async (message) => {
  if (message) {
    // Load model or infer depends on the first data
    switch (message.data.action) {
      case 'load': {
        if (modelRunner) {
          modelRunner.delete();
        }
        const loadStart = performance.now();
        const modelPath = message.data.modelPath;
        // Load WASM module and model.
        const [module, modelArrayBuffer] = await Promise.all([
          tflite_model_runner_ModuleFactory(),
          (await fetch(modelPath)).arrayBuffer(),
        ]);
        // Load WASM module and model.
        const modelBytes = new Uint8Array(modelArrayBuffer);
        const offset = module._malloc(modelBytes.length);
        module.HEAPU8.set(modelBytes, offset);

        // Create model runner.
        modelRunnerResult =
          module.TFLiteWebModelRunner.CreateFromBufferAndOptions(
              offset,
              modelBytes.length,
              {
                numThreads: 1,
                enableWebNNDelegate: message.data.enableWebNNDelegate,
                webNNDevicePreference: parseInt(message.data.webNNDevicePreference),
              },
          );

        if (!modelRunnerResult.ok()) {
          throw new Error(
              `Failed to create TFLiteWebModelRunner: ${modelRunner.errorMessage()}`);
        }
        modelRunner = modelRunnerResult.value();
        const loadFinishedMs = (performance.now() - loadStart).toFixed(2);
        postMessage(loadFinishedMs);
        break;
      }
      case 'compute': {
        // Get input and output info.

        const inputs = callAndDelete(modelRunner.GetInputs(), (results) => convertCppVectorToArray(results));
        const input = inputs[0];
        const outputs = callAndDelete(modelRunner.GetOutputs(), (results) => convertCppVectorToArray(results));
        const output = outputs[0];

        // Set input tensor data from the image (224 x 224 x 3).
        const inputBuffer = input.data();
        inputBuffer.set(message.data.buffer);

        // Infer, get output tensor, and sort by logit values in reverse.
        const inferStart = performance.now();
        modelRunner.Infer();
        const inferTime = performance.now() - inferStart;
        console.log(`Infer time in worker: ${inferTime.toFixed(2)} ms`);

        let outputBuffer = output.data();
        outputBuffer = outputBuffer.slice(0);
        postMessage({outputBuffer}, [outputBuffer.buffer]);
        break;
      }
      default: {
        break;
      }
    }
  }
};

// Helper functions.

// Converts the given c++ vector to a JS array.
function convertCppVectorToArray(vector) {
  if (vector == null) return [];

  const result = [];
  for (let i = 0; i < vector.size(); i++) {
    const item = vector.get(i);
    result.push(item);
  }
  return result;
}


// Calls the given function with the given deletable argument, ensuring that
// the argument gets deleted afterwards (even if the function throws an error).
function callAndDelete(arg, func) {
  try {
    return func(arg);
  } finally {
    if (arg != null) arg.delete();
  }
}
