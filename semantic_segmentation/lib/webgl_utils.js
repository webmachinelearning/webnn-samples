export class WebGLUtils {
  constructor(context) {
    this.gl_ = context;
    this.fbo_ = {};
    this.tex_ = {};
  }

  setTexture(name, tex) {
    this.tex_[name] = tex;
  }

  getTexture(name) {
    return this.tex_[name];
  }

  setup2dQuad() {
    const quad = new Float32Array([-1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1]);
    const vbo = this.gl_.createBuffer();
    this.gl_.bindBuffer(this.gl_.ARRAY_BUFFER, vbo);
    this.gl_.bufferData(this.gl_.ARRAY_BUFFER, quad, this.gl_.STATIC_DRAW);
    this.gl_.enableVertexAttribArray(0);
    this.gl_.vertexAttribPointer(0, 2, this.gl_.FLOAT, false, 0, 0);
  }

  createAndBindTexture(params) {
    const name = params.name;
    const filter = params.filter || this.gl_.LINEAR;
    const target = params.target || this.gl_.TEXTURE_2D;
    const level = params.level || 0;
    const width = params.width || 1;
    const height = params.height || 1;
    const border = params.border || 0;
    const format = params.format || this.gl_.RGBA;
    const internalformat = params.internalformat || format;
    const type = params.type || this.gl_.UNSIGNED_BYTE;
    const source = params.format || null;

    const texture = this.gl_.createTexture();

    this.gl_.bindTexture(this.gl_.TEXTURE_2D, texture);
    this.gl_.texParameteri(
        this.gl_.TEXTURE_2D, this.gl_.TEXTURE_WRAP_S, this.gl_.CLAMP_TO_EDGE);
    this.gl_.texParameteri(
        this.gl_.TEXTURE_2D, this.gl_.TEXTURE_WRAP_T, this.gl_.CLAMP_TO_EDGE);
    this.gl_.texParameteri(
        this.gl_.TEXTURE_2D, this.gl_.TEXTURE_MIN_FILTER, filter);
    this.gl_.texParameteri(
        this.gl_.TEXTURE_2D, this.gl_.TEXTURE_MAG_FILTER, filter);
    this.gl_.texImage2D(target, level, internalformat, width, height, border,
        format, type, source);

    this.tex_[name] = texture;

    return texture;
  }

  createTextures(textures) {
    for (const tex of textures) {
      this.createAndBindTexture(tex);
    }
  }

  bindInputTextures(textures) {
    for (let i = 0; i < textures.length; i++) {
      this.gl_.activeTexture(this.gl_.TEXTURE0 + i);
      this.gl_.bindTexture(this.gl_.TEXTURE_2D, this.tex_[textures[i]]);
    }
  }

  bindFramebuffer(fboName) {
    if (fboName === null) {
      this.gl_.bindFramebuffer(this.gl_.FRAMEBUFFER, null);
    } else {
      this.gl_.bindFramebuffer(this.gl_.FRAMEBUFFER, this.fbo_[fboName]);
    }
  }

  bindTexture(texName) {
    this.gl_.bindTexture(this.gl_.TEXTURE_2D, this.tex_[texName]);
  }

  createTexInFrameBuffer(fboName, texturesParams) {
    const newFbo = this.gl_.createFramebuffer();
    this.gl_.bindFramebuffer(this.gl_.FRAMEBUFFER, newFbo);
    this.fbo_[fboName] = newFbo;

    const attachments = [];

    for (let i = 0; i < texturesParams.length; i++) {
      const params = texturesParams[i];
      const attach = (params.attach || i) + this.gl_.COLOR_ATTACHMENT0;
      attachments.push(attach);

      const newTex = this.createAndBindTexture(params);

      this.gl_.framebufferTexture2D(
          this.gl_.FRAMEBUFFER, attach, this.gl_.TEXTURE_2D, newTex, 0);
    }

    if (attachments.length > 1) {
      this.gl_.drawBuffers(attachments);
    }

    const status = this.gl_.checkFramebufferStatus(this.gl_.FRAMEBUFFER);
    if (status !== this.gl_.FRAMEBUFFER_COMPLETE) {
      console.warn('FBO is not complete');
    }

    this.gl_.bindFramebuffer(this.gl_.FRAMEBUFFER, null);
  }

  setViewport(width, height) {
    this.gl_.viewport(0, 0, width, height);
  }

  render() {
    this.gl_.drawArrays(this.gl_.TRIANGLES, 0, 6);
  }

  delete() {
    this.fbo_ = {};
    this.tex_ = {};
  }
}
