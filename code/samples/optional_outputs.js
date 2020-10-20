const nn = navigator.ml.getNeuralNetworkContext();

// Build a model with two outputs.
const builder = nn.createModelBuilder();
const descA = {type: 'float32', dimensions: [3, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dimensions: [4, 3]};
const bufferB = new Float32Array(sizeOfShape(descB.dimensions)).fill(0.5);
const b = builder.constant(descB, bufferB);
const descC = {type: 'float32', dimensions: [3, 3]};
const bufferC = new Float32Array(sizeOfShape(descC.dimensions)).fill(1);
const c = builder.constant(descC, bufferC);
const d = builder.matmul(a, b);
const e = builder.add(d, c);
const model = builder.createModel({d, e});

const compiledModel = await model.compile();
const bufferA = new Float32Array(sizeOfShape(descA.dimensions)).fill(0.5);
const inputs = {'a': {buffer: bufferA}};

// Compute both d and e.
let outputs = await compiledModel.compute(inputs);
console.log(`outputs include ${Object.keys(outputs)}`);

// Compute d.
outputs = await compiledModel.compute(inputs, {d});
console.log(`outputs include ${Object.keys(outputs)}`);
console.log(`shape: [${outputs.d.dimensions}], values: ${outputs.d.buffer}`);

// Compute e.
outputs = await compiledModel.compute(inputs, {e});
console.log(`outputs include ${Object.keys(outputs)}`);
console.log(`shape: [${outputs.e.dimensions}], values: ${outputs.e.buffer}`);
