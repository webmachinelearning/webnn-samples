const nn = navigator.ml.getNeuralNetworkContext();

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

const builder = nn.createModelBuilder();

// Create OperandDescriptor object.
const float32TensorType = {type: 'float32', dimensions: TENSOR_DIMS};

// constant1 is a constant operand with the value 0.5.
const constantBuffer1 = new Float32Array(TENSOR_SIZE).fill(0.5);
const constant1 = builder.constant(float32TensorType, constantBuffer1);

// input1 is one of the input operands. Its value will be set before execution.
const input1 = builder.input('input1', float32TensorType);

// constant2 is another constant operand with the value 0.5.
const constantBuffer2 = new Float32Array(TENSOR_SIZE).fill(0.5);
const constant2 = builder.constant(float32TensorType, constantBuffer2);

// input2 is another input operand. Its value will be set before execution.
const input2 = builder.input('input2', float32TensorType);

// intermediateOutput1 is the output of the first Add operation.
const intermediateOutput1 = builder.add(constant1, input1);

// intermediateOutput2 is the output of the second Add operation.
const intermediateOutput2 = builder.add(constant2, input2);

// output is the output operand of the Mul operation.
const output = builder.mul(intermediateOutput1, intermediateOutput2);

// Create the model by identifying the outputs.
const model = builder.createModel({'output': output});

// Compile the constructed model.
const compilation = await model.compile({powerPreference: 'low-power'});

// Setup the input buffers with value 1.
const inputBuffer1 = new Float32Array(TENSOR_SIZE).fill(1);
const inputBuffer2 = new Float32Array(TENSOR_SIZE).fill(1);

// Asynchronously execute the compiled model with the specified inputs.
const inputs = {
  'input1': {buffer: inputBuffer1},
  'input2': {buffer: inputBuffer2},
};
const outputs = await compilation.compute(inputs);

// The computed result in the output operand’s buffer.
console.log('Output shape: ' + outputs.output.dimensions);
console.log('Output value: ' + outputs.output.buffer);
