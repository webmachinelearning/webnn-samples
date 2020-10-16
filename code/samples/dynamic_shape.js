const nn = navigator.ml.getNeuralNetworkContext();

// Create a model with dynamic shaped inputs.
const builder = nn.createModelBuilder();
const descA = {type: 'float32', dimensions: [-1, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dimensions: [4, -1]};
const b = builder.input('b', descB);
const c = builder.matmul(a, b);
const model = builder.createModel({c});

const compiledModel = await model.compile();

async function compute(shapeA, shapeB) {
  const bufferA = new Float32Array(sizeOfShape(shapeA)).fill(0.5);
  const bufferB = new Float32Array(sizeOfShape(shapeB)).fill(0.5);

  // Specify the shape of inputs when computing.
  const inputs = {
    'a': {buffer: bufferA, dimensions: shapeA},
    'b': {buffer: bufferB, dimensions: shapeB},
  };
  const outputs = await compiledModel.compute(inputs);
  console.log(`shape: [${outputs.c.dimensions}], values: ${outputs.c.buffer}`);
}

await compute([3, 4], [4, 3]);
await compute([4, 4], [4, 4]);
await compute([5, 4], [4, 5]);
