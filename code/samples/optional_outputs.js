const context = await navigator.ml.createContext();

// Build a graph with two outputs.
const builder = new MLGraphBuilder(context);
const descA = {type: 'float32', dataType: 'float32', dimensions: [3, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dataType: 'float32', dimensions: [4, 3]};
const bufferB = new Float32Array(sizeOfShape(descB.dimensions)).fill(0.5);
const b = builder.constant(descB, bufferB);
const descC = {type: 'float32', dataType: 'float32', dimensions: [3, 3]};
const bufferC = new Float32Array(sizeOfShape(descC.dimensions)).fill(1);
const c = builder.constant(descC, bufferC);
const d = builder.matmul(a, b);
const e = builder.add(d, c);
const graph = await builder.build({'d': d, 'e': e});

const bufferA = new Float32Array(sizeOfShape(descA.dimensions)).fill(0.5);
const inputs = {'a': bufferA};

// Compute d.
const bufferD = new Float32Array(sizeOfShape([3, 3]));
const resultsD = await context.compute(graph, inputs, {'d': bufferD});
console.log(`values: ${resultsD.outputs.d}`);

// Compute e.
const bufferE = new Float32Array(sizeOfShape([3, 3]));
const resultsE = await context.compute(graph, inputs, {'e': bufferE});
console.log(`values: ${resultsE.outputs.e}`);
