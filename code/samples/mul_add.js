const desc = {dataType: 'float32', dimensions: [2, 2], shape: [2, 2]};
const context = await navigator.ml.createContext();
const builder = new MLGraphBuilder(context);
// 1. Create a computational graph 'C = 0.2 * A + B'.
const constant = builder.constant(
    {dataType: 'float32'}, new Float32Array([0.2]));
const A = builder.input('A', desc);
const B = builder.input('B', desc);
const C = builder.add(builder.mul(A, constant), B);
// 2. Build the graph into an executable.
const graph = await builder.build({'C': C});
// 3. Bind inputs to the graph and execute for the result.
const bufferA = new Float32Array(4).fill(1.0);
const bufferB = new Float32Array(4).fill(0.8);
desc.usage = MLTensorUsage.WRITE;
const tensorA = await context.createTensor(desc);
const tensorB = await context.createTensor(desc);
context.writeTensor(tensorA, bufferA);
context.writeTensor(tensorB, bufferB);
const tensorC = await context.createTensor({
  ...desc,
  usage: MLTensorUsage.READ,
});
const inputs = {'A': tensorA, 'B': tensorB};
const outputs = {'C': tensorC};
context.dispatch(graph, inputs, outputs);
// The computed result of [[1, 1], [1, 1]] is in the buffer associated with
// the output operand.
const results = await context.readTensor(tensorC);
console.log('Output value: ' + new Float32Array(results));
