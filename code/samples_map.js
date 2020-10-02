import {simpleGraph} from './samples/simple_graph.js';
import {matmul} from './samples/matmul.js';

const samples = [simpleGraph, matmul];

export const samplesMap = new Map();

for (const sample of samples) {
  samplesMap.set(sample.name, sample.code);
}
