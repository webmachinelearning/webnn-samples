// Step 0: Create a context and graph builder for 'gpu', 'cpu' or 'npu'.
const context = await navigator.ml.createContext({deviceType: 'gpu'});
const builder = new MLGraphBuilder(context);
// Step 1: Create a computational graph calculating `c = a * b`.
const a = builder.input('a', {
  dataType: 'float32',
  dimensions: [3, 4],
  shape: [3, 4],
});
const b = builder.input('b', {
  dataType: 'float32',
  dimensions: [4, 3],
  shape: [4, 3],
});
const c = builder.matmul(a, b);
// Step 2: Compile it into an executable graph.
const graph = await builder.build({c});
// Step 3: Bind input and output buffers to the graph and execute.
const bufferA = new Float32Array(3*4).fill(1.0);
const bufferB = new Float32Array(4*3).fill(0.8);
const bufferC = new Float32Array(3*3);
const results = await context.compute(
    graph, {'a': bufferA, 'b': bufferB}, {'c': bufferC});
// Step 4: Retrieve the results.
console.log(`values: ${results.outputs.c}`);
