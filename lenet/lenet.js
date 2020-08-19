
var nn = navigator.ml.getNeuralNetworkContext('v2');

function sizeOfShape(shape) {
  return shape.reduce((a, b) => { return a * b; });
}

class Lenet {
  constructor(url) {
    this.url_ = url;
    this.model_ = null;
    this.compilation_ = null;
  }

  async load() {
    const response = await fetch(this.url_);
    const arrayBuffer = await response.arrayBuffer();
    if (arrayBuffer.byteLength !== 1724336) {
      throw new Error('Incorrect weights file');
    }

    const inputShape = [1, 1, 28, 28];
    const input = nn.input('input', {type: 'tensor-float32', dimensions: inputShape});

    const conv1FitlerShape = [20, 1, 5, 5];
    let byteOffset = 0;
    const conv1FilterData = new Float32Array(arrayBuffer, byteOffset, sizeOfShape(conv1FitlerShape));
    const conv1Filter = nn.constant({type: 'tensor-float32', dimensions: conv1FitlerShape},
                                    conv1FilterData);
    byteOffset += sizeOfShape(conv1FitlerShape) * Float32Array.BYTES_PER_ELEMENT;
    const conv1 = nn.conv2d(input, conv1Filter);

    const add1BiasShape = [1, 20, 1, 1];
    const add1BiasData = new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add1BiasShape));
    const add1Bias = nn.constant({type: 'tensor-float32', dimensions: add1BiasShape},
                                 add1BiasData);
    byteOffset += sizeOfShape(add1BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add1 = nn.add(conv1, add1Bias);

    const pool1WindowShape = [2, 2];
    const pool1Strides = [2, 2];
    const pool1 = nn.maxPool2d(add1, pool1WindowShape, [0, 0, 0, 0], pool1Strides);

    const conv2FilterShape = [50, 20, 5, 5];
    const conv2Filter = nn.constant({type: 'tensor-float32', dimensions: conv2FilterShape},
                                    new Float32Array(arrayBuffer, byteOffset, sizeOfShape(conv2FilterShape)));
    byteOffset += sizeOfShape(conv2FilterShape) * Float32Array.BYTES_PER_ELEMENT;
    const conv2 = nn.conv2d(pool1, conv2Filter);

    const add2BiasShape = [1, 50, 1, 1];
    const add2Bias = nn.constant({type: 'tensor-float32', dimensions: add2BiasShape},
                                 new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add2BiasShape)));
    byteOffset += sizeOfShape(add2BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add2 = nn.add(conv2, add2Bias);

    const pool2WindowShape = [2, 2];
    const pool2Strides = [2, 2];
    const pool2 = nn.maxPool2d(add2, pool2WindowShape, [0, 0, 0, 0], pool2Strides);

    const reshape1Shape = [1, -1];
    const reshape1 = nn.reshape(pool2, reshape1Shape);

    // skip the new shape
    byteOffset += 2 * BigInt64Array.BYTES_PER_ELEMENT;

    const matmul1Shape = [500, 800];
    const matmul1Weights = nn.constant({type: 'tensor-float32', dimensions: matmul1Shape},
                                       new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul1Shape)));
    byteOffset += sizeOfShape(matmul1Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul1WeightsTransposed = nn.transpose(matmul1Weights);
    const matmul1 = nn.matmul(reshape1, matmul1WeightsTransposed);

    const add3BiasShape = [1, 500];
    const add3Bias = nn.constant({type: 'tensor-float32', dimensions: add3BiasShape},
                                 new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add3BiasShape)));
    byteOffset += sizeOfShape(add3BiasShape) * Float32Array.BYTES_PER_ELEMENT;
    const add3 = nn.add(matmul1, add3Bias);

    const relu = nn.relu(add3);

    const reshape2Shape = [1, -1];
    const reshape2 = nn.reshape(relu, reshape2Shape);

    const matmul2Shape = [10, 500];
    const matmul2Weights = nn.constant({type: 'tensor-float32', dimensions: matmul2Shape},
                                       new Float32Array(arrayBuffer, byteOffset, sizeOfShape(matmul2Shape)));
    byteOffset += sizeOfShape(matmul2Shape) * Float32Array.BYTES_PER_ELEMENT;
    const matmul2WeightsTransposed = nn.transpose(matmul2Weights);
    const matmul2 = nn.matmul(reshape2, matmul2WeightsTransposed);

    const add4BiasShape = [1, 10];
    const add4Bias = nn.constant({type: 'tensor-float32', dimensions: add4BiasShape},
                                 new Float32Array(arrayBuffer, byteOffset, sizeOfShape(add4BiasShape)));
    const add4 = nn.add(matmul2, add4Bias);

    const softmax = nn.softmax(add4)

    this.model_ = await nn.createModel([{name: 'output', operand: softmax}]);
  }

  async compile(options) {
    this.compilation_ = await this.model_.createCompilation(options);
  }

  async predict(inputBuffer) {
    const outputBuffer = new Float32Array(10);
    const execution = await this.compilation_.createExecution();
    execution.setInput('input', inputBuffer);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    return Array.from(outputBuffer);
  }
}
