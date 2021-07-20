import {WebGLUtils} from './webgl_utils.js';
import {Shader} from './shader.js';

/* eslint max-len: ["error", {"code": 95}] */

export class GuidedFilter {
  constructor(context) {
    this.gl = context;
    this.utils = new WebGLUtils(context);

    if (!this.gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('not support EXT_color_buffer_float');
    }

    if (!this.gl.getExtension('OES_texture_float_linear')) {
      throw new Error('not support OES_texture_float_linear');
    }

    this.shaders = {};

    this.width = 513;
    this.height = 513;
    this.subsample = 4;
    this.radius = 16;
    this.epsilon = 1e-6;
    if (this.radius < 4) {
      this.subRadius = this.radius;
      this.subWidth = this.width;
      this.subHeight = this.height;
    } else {
      this.subRadius = this.radius / this.subsample;
      this.subWidth = this.width / this.subsample;
      this.subHeight = this.height / this.subsample;
    }
  }

  setup(radius, epsilon, width, height) {
    this.radius = radius;
    this.epsilon = epsilon;
    this.width = width;
    this.height = height;

    if (this.radius < 4) {
      this.subRadius = this.radius;
      this.subWidth = this.width;
      this.subHeight = this.height;
    } else {
      this.subRadius = this.radius / this.subsample;
      this.subWidth = this.width / this.subsample;
      this.subHeight = this.height / this.subsample;
    }

    this.utils.setup2dQuad();
    this.setupHadamard4Shader_();
    this.setupBoxFilterShader_();
    this.setupHadamard2Shader_();
    this.setupCovarShader_();
    this.setupFinalShader_();
  }

  setupHadamard4Shader_() {
    const vs =
      `#version 300 es
        in vec4 a_pos;
        out vec2 v_texcoord;

        void main() {
          gl_Position = a_pos;
          v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
        }`;

    const fs =
      `#version 300 es
        precision highp float;

        out vec4 result;

        uniform sampler2D u_p;
        uniform sampler2D u_I;

        in vec2 v_texcoord;

        void main() {
          float I = texture(u_I, v_texcoord).x;
          float p = texture(u_p, v_texcoord).a;
          float Ip = I * p;
          float II = I * I;
          result = vec4(I, p, Ip, II);
        }`;

    this.shaders.hadamard4 = new Shader(this.gl, vs, fs);

    this.utils.createAndBindTexture({
      name: 'p',
      filter: this.gl.NEAREST,
    });

    this.utils.createAndBindTexture({
      name: 'I',
      filter: this.gl.LINEAR,
    });

    this.utils.createTexInFrameBuffer('fbo1',
        [{
          name: 'result1',
          width: this.subWidth,
          height: this.subHeight,
          filter: this.gl.NEAREST,
          type: this.gl.FLOAT,
          internalformat: this.gl.RGBA32F,
        }],
    );
    this.shaders.hadamard4.use();
    this.shaders.hadamard4.set1i('u_p', 0); // texture units 0
    this.shaders.hadamard4.set1i('u_I', 1); // texture units 1
  }

  setupBoxFilterShader_() {
    const vs =
      `#version 300 es
        in vec4 a_pos;
        out vec2 v_texcoord;

        void main() {
          gl_Position = a_pos;
          v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
        }`;

    const fs =
      `#version 300 es
        precision highp float;
        out vec4 out_color;
        in vec2 v_texcoord;

        uniform sampler2D bg;
        uniform bool first_pass;
        uniform int radius;

        void main() {
          vec2 tex_offset = 1.0 / vec2(textureSize(bg, 0));
          vec4 result = texture(bg, v_texcoord);
          if (first_pass) {
            for (int i = 1; i <= radius; ++i) {
              result += texture(bg, v_texcoord + vec2(tex_offset.x * float(i), 0.0));
              result += texture(bg, v_texcoord - vec2(tex_offset.x * float(i), 0.0));
            }
          } else {
            for (int i = 1; i <= radius; ++i) {
              result += texture(bg, v_texcoord + vec2(0.0, tex_offset.y * float(i)));
              result += texture(bg, v_texcoord - vec2(0.0, tex_offset.y * float(i)));
            }
          }
          out_color = result / (2.0 * float(radius) + 1.0);
        }`;

    this.shaders.boxFilter = new Shader(this.gl, vs, fs);
    this.utils.createTexInFrameBuffer('pingpong',
        [{
          name: 'pingpongTemp',
          width: this.subWidth,
          height: this.subHeight,
          filter: this.gl.LINEAR,
          type: this.gl.FLOAT,
          internalformat: this.gl.RGBA32F,
        }],
    );

    this.utils.createTexInFrameBuffer('fbo2',
        [{
          name: 'result2',
          width: this.subWidth,
          height: this.subHeight,
          filter: this.gl.LINEAR,
          type: this.gl.FLOAT,
          internalformat: this.gl.RGBA32F,
        }],
    );

    this.utils.createTexInFrameBuffer('fbo5',
        [{
          name: 'result5',
          width: this.width,
          height: this.height,
          filter: this.gl.LINEAR,
          type: this.gl.FLOAT,
          internalformat: this.gl.RGBA32F,
        }],
    );
  }


  setupHadamard2Shader_() {
    const vs =
      `#version 300 es
        in vec4 a_pos;
        out vec2 v_texcoord;

        void main() {
          gl_Position = a_pos;
          v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
        }`;

    const fs =
      `#version 300 es
        precision highp float;

        in vec2 v_texcoord;
        out vec4 result;

        uniform sampler2D result2;

        void main() {
          vec4 prev = texture(result2, v_texcoord);
          float mean_I = prev.x;
          float mean_p = prev.y;
          result.xy = vec2(mean_I * mean_p, mean_I * mean_I);
        }`;

    this.shaders.hadamard2 = new Shader(this.gl, vs, fs);

    this.utils.createTexInFrameBuffer('fbo3',
        [{
          name: 'result3',
          width: this.subWidth,
          height: this.subHeight,
          filter: this.gl.NEAREST,
          type: this.gl.FLOAT,
          internalformat: this.gl.RGBA32F,
        }],
    );
  }

  setupCovarShader_() {
    const vs =
      `#version 300 es
        in vec4 a_pos;
        out vec2 v_texcoord;

        void main() {
          gl_Position = a_pos;
          v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
        }`;

    const fs =
      `#version 300 es
        precision highp float;

        in vec2 v_texcoord;
        out vec4 result;

        uniform sampler2D result2;
        uniform sampler2D result3;
        uniform float epsilon;

        void main() {
          vec4 r2 = texture(result2, v_texcoord);
          float meanI = r2.x;
          float meanp = r2.y;
          float meanIp = r2.z;
          float meanII = r2.a;
          vec4 r3 = texture(result3, v_texcoord);
          float meanI_meanp = r3.x;
          float meanI_meanI = r3.y;

          // this is the covariance of (I, p) in each local patch
          float covIp = meanIp - meanI_meanp;
          float varI = meanII - meanI_meanI;
          float a = covIp / (varI + epsilon);
          float b = meanp - a * meanI;
          result.xy = vec2(a, b);
        }`;

    this.shaders.covar = new Shader(this.gl, vs, fs);

    this.shaders.covar.use();
    this.shaders.covar.set1i('result2', 0); // texture units 0
    this.shaders.covar.set1i('result3', 1); // texture units 1

    this.utils.createTexInFrameBuffer('fbo4',
        [{
          name: 'result4',
          width: this.subWidth,
          height: this.subHeight,
          filter: this.gl.NEAREST,
          type: this.gl.FLOAT,
          internalformat: this.gl.RGBA32F,
        }],
    );
  }

  setupFinalShader_() {
    const vs =
      `#version 300 es
        in vec4 a_pos;
        out vec2 v_texcoord;
        out vec2 v_flipcord;

        void main() {
          gl_Position = a_pos;
          v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
          v_flipcord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
        }`;

    const fs =
      `#version 300 es
        precision highp float;

        out vec4 result;

        uniform sampler2D result5;
        uniform sampler2D u_I;

        in vec2 v_flipcord;
        in vec2 v_texcoord;

        void main() {
          vec4 r5 = texture(result5, v_texcoord);
          vec4 I = texture(u_I, v_flipcord);
          float mean_a = r5.x;
          float mean_b = r5.y;

          // q = mean_a .* I + mean_b;
          float intensity = mean_a * I.z + mean_b;
          result = vec4(intensity, intensity, intensity, intensity);
        }`;

    this.shaders.final = new Shader(this.gl, vs, fs);

    this.shaders.final.use();
    this.shaders.final.set1i('result5', 0); // texture units 0
    this.shaders.final.set1i('u_I', 1); // texture units 1

    this.utils.createTexInFrameBuffer('fbo6',
        [{
          name: 'result6',
          width: this.width,
          height: this.height,
          filter: this.gl.LINEAR,
        }],
    );
  }

  apply(guideImage, input, maskWidth, maskHeight) {
    this.utils.bindTexture('p');
    this.gl.pixelStorei(this.gl.UNPACK_ALIGNMENT, 1);
    this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.ALPHA,
        maskWidth,
        maskHeight,
        0,
        this.gl.ALPHA,
        this.gl.UNSIGNED_BYTE,
        input,
    );

    this.utils.bindTexture('I');
    this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.LUMINANCE,
        this.gl.LUMINANCE,
        this.gl.UNSIGNED_BYTE,
        guideImage,
    );

    this.shaders.hadamard4.use();
    this.utils.bindFramebuffer('fbo1');
    this.utils.bindInputTextures(['p', 'I']);
    this.utils.setViewport(this.subWidth, this.subHeight);
    this.utils.render();

    this.shaders.boxFilter.use();
    this.shaders.boxFilter.set1i('first_pass', 1);
    this.shaders.boxFilter.set1i('radius', this.subRadius);
    this.utils.bindFramebuffer('pingpong');
    this.utils.bindInputTextures(['result1']);
    this.utils.render();

    this.shaders.boxFilter.set1i('first_pass', 0);
    this.utils.bindFramebuffer('fbo2');
    this.utils.bindInputTextures(['pingpongTemp']);
    this.utils.render();

    this.shaders.hadamard2.use();
    this.utils.bindFramebuffer('fbo3');
    this.utils.bindInputTextures(['result2']);
    this.utils.render();

    this.shaders.covar.use();
    this.shaders.covar.set1f('epsilon', this.epsilon);
    this.utils.bindFramebuffer('fbo4');
    this.utils.bindInputTextures(['result2', 'result3']);
    this.utils.render();

    this.shaders.boxFilter.use();
    this.shaders.boxFilter.set1i('first_pass', 1);
    this.shaders.boxFilter.set1i('radius', this.subRadius);
    this.utils.bindFramebuffer('pingpong');
    this.utils.bindInputTextures(['result4']);
    this.utils.render();

    this.shaders.boxFilter.set1i('first_pass', 0);
    this.utils.bindFramebuffer('fbo5');
    this.utils.bindInputTextures(['pingpongTemp']);
    this.utils.setViewport(this.width, this.height);
    this.utils.render();

    this.shaders.final.use();
    this.utils.bindFramebuffer('fbo6');
    this.utils.bindInputTextures(['result5', 'I']);
    this.utils.render();

    return this.utils.getTexture('result6');
  }
}
