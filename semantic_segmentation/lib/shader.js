export class Shader {
  constructor(gl, vertShaderSrc, fragShaderSrc) {
    this.gl_ = gl;
    this.loc_ = [];

    const vs = this.createShader_(this.gl_.VERTEX_SHADER, vertShaderSrc);
    const fs = this.createShader_(this.gl_.FRAGMENT_SHADER, fragShaderSrc);
    this.prog_ = this.createProgram_(vs, fs);

    // get location of all uniforms in vertex and fragment shader codes
    // note: this regex will take into account uniforms within comments
    const regex = /uniform\s+[^\s]+\s+([_a-zA-Z][_a-zA-Z0-9]*)[=[\s]?[^;]*;/g;
    const src = vertShaderSrc + fragShaderSrc;
    for (let match; match = regex.exec(src); ) {
      const uniform = match[1];
      // location will be null if uniform is commented or optimized out
      this.loc_[uniform] = this.gl_.getUniformLocation(this.prog_, uniform);
    }
  }

  use() {
    this.gl_.useProgram(this.prog_);
  }

  set1i(name, x) {
    this.gl_.uniform1i(this.loc_[name], x);
  }

  set4f(name, w, x, y, z) {
    this.gl_.uniform4f(this.loc_[name], w, x, y, z);
  }

  set2f(name, x, y) {
    this.gl_.uniform2f(this.loc_[name], x, y);
  }

  set1f(name, x) {
    this.gl_.uniform1f(this.loc_[name], x);
  }

  set1fv(name, arr) {
    this.gl_.uniform1fv(this.loc_[name], arr);
  }

  createShader_(type, source) {
    const shader = this.gl_.createShader(type);
    this.gl_.shaderSource(shader, source);
    this.gl_.compileShader(shader);
    if (!this.gl_.getShaderParameter(shader, this.gl_.COMPILE_STATUS)) {
      const log = this.gl_.getShaderInfoLog(shader);
      this.gl_.deleteShader(shader);
      throw new Error(log);
    }
    return shader;
  }

  createProgram_(vertexShader, fragmentShader) {
    const program = this.gl_.createProgram();
    this.gl_.attachShader(program, vertexShader);
    this.gl_.attachShader(program, fragmentShader);
    this.gl_.linkProgram(program);
    if (!this.gl_.getProgramParameter(program, this.gl_.LINK_STATUS)) {
      const log = this.gl_.getProgramInfoLog(program);
      this.gl_.deleteProgram(program);
      throw new Error(log);
    }
    return program;
  }
}
