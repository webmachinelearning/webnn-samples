export const simpleGraph = {name: 'simple graph', code:
`const nn = navigator.ml.getNeuralNetworkContext();

// The following code builds a graph as:
// constant1 ---+
//              +--- Add ---> intermediateOutput1 ---+
// input1    ---+                                    |
//                                                   +--- Mul---> output
// constant2 ---+                                    |
//              +--- Add ---> intermediateOutput2 ---+
// input2    ---+

// Use tensors in 4 dimensions.
const TENSOR_DIMS = [2, 2, 2, 2];
const TENSOR_SIZE = 16;

// Create OperandDescriptor object.
const float32TensorType = {type: 'tensor-float32', dimensions: TENSOR_DIMS};

// constant1 is a constant tensor with the value 0.5.
const constantBuffer1 = new Float32Array(TENSOR_SIZE).fill(0.5);
const constant1 = nn.constant(float32TensorType, constantBuffer1);

// input1 is one of the input tensors. Its value will be set before execution.
const input1 = nn.input('input1', float32TensorType);

// constant2 is another constant tensor with the value 0.5.
const constantBuffer2 = new Float32Array(TENSOR_SIZE).fill(0.5);
const constant2 = nn.constant(float32TensorType, constantBuffer2);

// input2 is another input tensor. Its value will be set before execution.
const input2 = nn.input('input2', float32TensorType);

// intermediateOutput1 is the output of the first Add operation.
const intermediateOutput1 = nn.add(constant1, input1);

// intermediateOutput2 is the output of the second Add operation.
const intermediateOutput2 = nn.add(constant2, input2);

// output is the output tensor of the Mul operation.
const output = nn.mul(intermediateOutput1, intermediateOutput2);

// Create the model by identifying the outputs.
const model = await nn.createModel([{name: 'output', operand: output}]);

// Create a Compilation object for the constructed model.
const options = { powerPreference: 'low-power' };
const compilation = await model.createCompilation(options);

// Create an Execution object for the compiled model.
const execution = await compilation.createExecution();

// Setup the input buffers with value 1.
const inputBuffer1 = new Float32Array(TENSOR_SIZE).fill(1);
const inputBuffer2 = new Float32Array(TENSOR_SIZE).fill(1);

// Associate the input buffers to model’s inputs.
execution.setInput('input1', inputBuffer1);
execution.setInput('input2', inputBuffer2);

// Associate the output buffer to model’s output.
let outputBuffer = new Float32Array(TENSOR_SIZE);
execution.setOutput('output', outputBuffer);

// Start the asynchronous computation.
await execution.startCompute();
// The computed result is now in outputBuffer.
console.log(outputBuffer);
`};