// The samples under ./samples folder
const samples = [
  'mul_add.js',
  'simple_graph.js',
  'matmul.js',
  'dynamic_shape.js',
  'optional_outputs.js',
];

class SamplesRepository {
  constructor(samples) {
    this.samples_ = new Map();
    for (const fileName of samples) {
      const url = './samples/' + fileName;
      this.samples_.set(fileName, {url});
    }
  }

  async getCode(name) {
    if (this.samples_.get(name).code === undefined) {
      const response = await fetch(this.samples_.get(name).url);
      const code = await response.text();
      this.samples_.get(name).code = code;
    }
    return this.samples_.get(name).code;
  }

  names() {
    return this.samples_.keys();
  }
}

export const samplesRepo = new SamplesRepository(samples);
