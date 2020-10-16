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

const shapeA = [3, 4];
const shapeB = [4, 3];
const bufferA = new Float32Array(sizeOfShape(shapeA)).fill(0.5);
const bufferB = new Float32Array(sizeOfShape(shapeB)).fill(0.5);

// Specify the shape of inputs when computing.
const inputs = {
  'a': {buffer: bufferA, dimensions: shapeA},
  'b': {buffer: bufferB, dimensions: shapeB},
};
const outputs = await compiledModel.compute(inputs);
console.log(`shape: [${outputs.c.dimensions}]`);
console.log(`values: ${outputs.c.buffer}`);
