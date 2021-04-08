const context = navigator.ml.createContext({powerPreference: 'low-power'});

// The following code builds a graph as:
// constant1 ---+
//              +--- Add ---> intermediateOutput1 ---+
// input1    ---+                                    |
//                                                   +--- Mul---> output
// constant2 ---+                                    |
//              +--- Add ---> intermediateOutput2 ---+
// input2    ---+

// Use tensors in 4 dimensions.
const TENSOR_DIMS = [1, 2, 2, 2];
const TENSOR_SIZE = 8;

const builder = new MLGraphBuilder(context);

// Create OperandDescriptor object.
const desc = {type: 'float32', dimensions: TENSOR_DIMS};

// constant1 is a constant operand with the value 0.5.
const constantBuffer1 = new Float32Array(TENSOR_SIZE).fill(0.5);
const constant1 = builder.constant(desc, constantBuffer1);

// input1 is one of the input operands. Its value will be set before execution.
const input1 = builder.input('input1', desc);

// constant2 is another constant operand with the value 0.5.
const constantBuffer2 = new Float32Array(TENSOR_SIZE).fill(0.5);
const constant2 = builder.constant(desc, constantBuffer2);

// input2 is another input operand. Its value will be set before execution.
const input2 = builder.input('input2', desc);

// intermediateOutput1 is the output of the first Add operation.
const intermediateOutput1 = builder.add(constant1, input1);

// intermediateOutput2 is the output of the second Add operation.
const intermediateOutput2 = builder.add(constant2, input2);

// output is the output operand of the Mul operation.
const output = builder.mul(intermediateOutput1, intermediateOutput2);

// Build graph.
const graph = await builder.build({'output': output});

// Setup the input buffers with value 1.
const inputBuffer1 = new Float32Array(TENSOR_SIZE).fill(1);
const inputBuffer2 = new Float32Array(TENSOR_SIZE).fill(1);

// Asynchronously execute the built model with the specified inputs.
const inputs = {
  'input1': {data: inputBuffer1},
  'input2': {data: inputBuffer2},
};
const outputs = await graph.compute(inputs);

// Log the shape and computed result of the output operand.
console.log('Output shape: ' + outputs.output.dimensions);
// Output shape: 1,2,2,2
console.log('Output value: ' + outputs.output.data);
// Output value: 2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25
