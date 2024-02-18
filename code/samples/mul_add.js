const operandType = {dataType: 'float32', dimensions: [2, 2]};
const context = await navigator.ml.createContext();
const builder = new MLGraphBuilder(context);
// 1. Create a computational graph 'C = 0.2 * A + B'.
const constant = builder.constant(
    {dataType: 'float32'}, new Float32Array([0.2]));
const A = builder.input('A', operandType);
const B = builder.input('B', operandType);
const C = builder.add(builder.mul(A, constant), B);
// 2. Build the graph into an executable.
const graph = await builder.build({'C': C});
// 3. Bind inputs to the graph and execute for the result.
const bufferA = new Float32Array(4).fill(1.0);
const bufferB = new Float32Array(4).fill(0.8);
const bufferC = new Float32Array(4);
const inputs = {'A': bufferA, 'B': bufferB};
const outputs = {'C': bufferC};
const results = await context.compute(graph, inputs, outputs);
// The computed result of [[1, 1], [1, 1]] is in the buffer associated with
// the output operand.
console.log('Output value: ' + results.outputs.C);
