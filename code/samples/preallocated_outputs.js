const nn = navigator.ml.getNeuralNetworkContext();

// The following code multiplies matrix a [3, 4] with matrix b [4, 3]
// into matrix c [3, 3].
const builder = nn.createModelBuilder();
const descA = {type: 'float32', dimensions: [3, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dimensions: [4, 3]};
const bufferB = new Float32Array(sizeOfShape(descB.dimensions)).fill(0.5);
const b = builder.constant(descB, bufferB);
const c = builder.matmul(a, b);
const model = builder.createModel({c});

const compiledModel = await model.compile();
const bufferA = new Float32Array(sizeOfShape(descA.dimensions)).fill(0.5);
const inputs = {'a': {buffer: bufferA}};
// Pre-allocate output buffer for c.
const outputs = {'c': {buffer: new Float32Array(sizeOfShape([3, 3]))}};
await compiledModel.compute(inputs, outputs);
console.log(`values: ${outputs.c.buffer}`);
