const context = navigator.ml.createContext();

// The following code multiplies matrix a [3, 4] with matrix b [4, 3]
// into matrix c [3, 3].
const builder = new MLGraphBuilder(context);
const descA = {type: 'float32', dimensions: [3, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dimensions: [4, 3]};
const bufferB = new Float32Array(sizeOfShape(descB.dimensions)).fill(0.5);
const b = builder.constant(descB, bufferB);
const c = builder.matmul(a, b);

const graph = builder.build({c});
const bufferA = new Float32Array(sizeOfShape(descA.dimensions)).fill(0.5);
const bufferC = new Float32Array(sizeOfShape([3, 3]));
const inputs = {'a': bufferA};
const outputs = {'c': bufferC};
graph.compute(inputs, outputs);
console.log(`values: ${bufferC}`);
