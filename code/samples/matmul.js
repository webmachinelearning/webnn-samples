export const matmul = {name: 'matmul', code:
`const nn = navigator.ml.getNeuralNetworkContext();

const descA = {type: 'tensor-float32', dimensions: [3, 4]};
const a = nn.input('a', descA);
const descB = {type: 'tensor-float32', dimensions: [4, 3]};
const b = nn.constant(descB, new Float32Array(sizeOfShape(descB.dimensions)).fill(0.5));
const c = nn.matmul(a, b);

const model = await nn.createModel([{name: 'c', operand: c}]);

const compilation = await model.createCompilation();

const execution = await compilation.createExecution();

const bufferA = new Float32Array(sizeOfShape(descA.dimensions)).fill(0.5);
execution.setInput('a', bufferA);

const bufferC = new Float32Array(sizeOfShape([descA.dimensions[0], descB.dimensions[1]]));
execution.setOutput('c', bufferC);

await execution.startCompute();
console.log(bufferC);
`};