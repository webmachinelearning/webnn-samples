// Step 0: Create a context and graph builder for 'gpu', 'cpu' or 'npu'.
const context = await navigator.ml.createContext({deviceType: 'gpu'});
const builder = new MLGraphBuilder(context);
const descA = {dataType: 'float32', dimensions: [3, 4], shape: [3, 4]};
const descB = {dataType: 'float32', dimensions: [4, 3], shape: [4, 3]};
// Step 1: Create a computational graph calculating `c = a * b`.
const a = builder.input('a', descA);
const b = builder.input('b', descB);
const c = builder.matmul(a, b);
// Step 2: Compile it into an executable graph.
const graph = await builder.build({c});
// Step 3: Bind input and output buffers to the graph and execute.
const bufferA = new Float32Array(3*4).fill(1.0);
const bufferB = new Float32Array(4*3).fill(0.8);
descA.usage = MLTensorUsage.WRITE;
descB.usage = MLTensorUsage.WRITE;
const tensorA = await context.createTensor(descA);
const tensorB = await context.createTensor(descB);
context.writeTensor(tensorA, bufferA);
context.writeTensor(tensorB, bufferB);
const tensorC = await context.createTensor({
  dataType: 'float32',
  dimensions: [3, 3],
  shape: [3, 3],
  usage: MLTensorUsage.READ,
});
context.dispatch(graph, {a: tensorA, b: tensorB}, {c: tensorC});
const results = await context.readTensor(tensorC);
// Step 4: Retrieve the results.
console.log(`values: ${new Float32Array(results)}`);
