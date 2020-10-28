const operandType = {type: 'float32', dimensions: [2, 2]};
const context = navigator.ml.getNeuralNetworkContext();
const builder = context.createModelBuilder();
// 1. Create a model of the computational graph 'C = 0.2 * A + B'.
const constant = builder.constant(0.2);
const A = builder.input('A', operandType);
const B = builder.input('B', operandType);
const C = builder.add(builder.mul(A, constant), B);
const model = builder.createModel({'C': C});
// 2. Compile the model into executable.
const compilation = await model.compile();
// 3. Bind inputs to the model and execute for the result.
const bufferA = new Float32Array(4).fill(1.0);
const bufferB = new Float32Array(4).fill(0.8);
const inputs = {'A': {buffer: bufferA}, 'B': {buffer: bufferB}};
const outputs = await compilation.compute(inputs);
// The computed result of [[1, 1], [1, 1]] is in the buffer associated with
// the output operand.
console.log('Output shape: ' + outputs.C.dimensions);
console.log('Output value: ' + outputs.C.buffer);
