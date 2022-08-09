const context = await navigator.ml.createContext();

// Create a graph with dynamic shaped inputs.
const builder = new MLGraphBuilder(context);
const descA = {type: 'float32', dimensions: [-1, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dimensions: [4, -1]};
const b = builder.input('b', descB);
const c = builder.matmul(a, b);
const graph = await builder.build({'c': c});

const compute = async (shapeA, shapeB, shapeC) => {
  const bufferA = new Float32Array(sizeOfShape(shapeA)).fill(0.5);
  const bufferB = new Float32Array(sizeOfShape(shapeB)).fill(0.5);
  const bufferC = new Float32Array(sizeOfShape(shapeC));

  // Specify the shape of inputs when computing.
  const inputs = {
    'a': {resource: bufferA, dimensions: shapeA},
    'b': {resource: bufferB, dimensions: shapeB},
  };
  const outputs = {'c': bufferC};
  await context.compute(graph, inputs, outputs);
  console.log(`values: ${bufferC}`);
};

await compute([3, 4], [4, 3], [3, 3]);
await compute([4, 4], [4, 4], [4, 4]);
await compute([5, 4], [4, 5], [5, 5]);
