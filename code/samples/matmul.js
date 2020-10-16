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
const outputs = await compiledModel.compute(inputs);
console.log(`shape: [${outputs.c.dimensions}]`);
console.log(`values: ${outputs.c.buffer}`);
