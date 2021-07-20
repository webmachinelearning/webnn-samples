import {GuidedFilter} from './guided_filter.js';
import {WebGLUtils} from './webgl_utils.js';
import {Shader} from './shader.js';

/* eslint max-len: ["error", {"code": 100}] */

export class Renderer {
  constructor(canvas) {
    this.gl = canvas.getContext('webgl2');
    if (this.gl === null) {
      throw new Error('Unable to initialize WebGL.');
    }

    this.guidedFilter = new GuidedFilter(this.gl);
    this.utils = new WebGLUtils(this.gl);

    this.shaders = {};

    this.segMap_ = null;
    this.predictions_ = null;
    this.fixedSize_ = 513;
    this.clippedSize_ = [this.fixedSize_, this.fixedSize_];
    this.imageSource_ = null;

    // UI state
    this.effect_ = 'label';
    this.zoom_ = 1;
    this.bgColor_ = [57, 135, 189];
    this.colorMapAlpha_ = 0.7;
    this.blurRadius_ = 30;
    this.backgroundImageSource_ = null;
    const kernel1D = this.generateGaussianKernel1D_(this.blurRadius_ * 2 + 1);
    this.halfKernel_ = kernel1D.slice(this.blurRadius_); // take the second half
    this.guidedFilterRadius_ = 0;

    this.colorPalette_ = new Uint8Array([
      45, 52, 54,
      85, 239, 196,
      129, 236, 236,
      116, 185, 255,
      162, 155, 254,
      223, 230, 233,
      0, 184, 148,
      0, 206, 201,
      9, 132, 227,
      39, 60, 117,
      108, 92, 231,
      178, 190, 195,
      255, 234, 167,
      250, 177, 160,
      255, 118, 117,
      253, 121, 168,
      99, 110, 114,
      253, 203, 110,
      225, 112, 85,
      214, 48, 49,
      232, 67, 147,
    ]);
  }

  get bgColor() {
    return this.bgColor_;
  }

  set bgColor(rgb) {
    this.bgColor_ = rgb;
    this.setupFillShader_();
    this.drawOutputs(this.segMap_);
  }

  get blurRadius() {
    return this.blurRadius_;
  }

  set blurRadius(radius) {
    if (this.blurRadius_ === radius) return;
    this.blurRadius_ = radius;
    const kernelSize = radius * 2 + 1;
    const kernel1D = this.generateGaussianKernel1D_(kernelSize, 25);
    // take the second half
    this.halfKernel_ = kernel1D.slice(radius);

    this.setupBlurShader_();
    this.drawOutputs(this.segMap_);
  }

  get colorMapAlpha() {
    return this.colorMapAlpha_;
  }

  set colorMapAlpha(alpha) {
    this.colorMapAlpha_ = alpha;

    this.drawColorLabel_();
  }

  get effect() {
    return this.effect_;
  }

  set effect(val) {
    this.effect_ = val;

    // shaders need to be rewired when changing effect
    this.setup();
    this.drawOutputs(this.segMap_);
  }

  get refineEdgeRadius() {
    return this.guidedFilterRadius_;
  }

  set refineEdgeRadius(radius) {
    this.guidedFilterRadius_ = radius;

    this.guidedFilter.setup(
        this.guidedFilterRadius_,
        1e-6,
        this.clippedSize_[0] * this.zoom_,
        this.clippedSize_[1] * this.zoom_,
    );
    this.setup();
    this.drawOutputs(this.segMap_);
  }

  get backgroundImageSource() {
    return this.backgroundImageSource_;
  }

  set backgroundImageSource(src) {
    this.backgroundImageSource_ = src;

    this.setupImageShader_();
    this.drawOutputs(this.segMap_);
  }

  setup() {
    //
    // I. Display color labels
    //
    // +------+                                    ============  +---------+
    // |  im  |--.                                 |  Shader  |  | Texture |
    // +------+  |  ============  +-----+          ============  +---------+
    //           |->| colorize |->| out |
    // +------+  |  |  Shader  |  +-----+
    // | pred |--'  ============
    // +------+
    //
    //
    // II. Person segmentation
    //
    // +------+                     +----+
    // |  im  |--.               .->| fg |----------------------------.
    // +------+  |  ===========  |  +----+                            |  =========  +-----+
    //           |->| extract |--|                                    |->| blend |->| out |
    // +------+  |  ===========  |  +----+  ========                  |  =========  +-----+
    // | pred |--'               '->| bg |->| blur |->|\              |
    // +------+                     +----+  ========  | |             |
    //                                      ========  | |  +--------+ |
    //                                      | fill |->| |->| styled |-'
    //                                      ========  | |  |   Bg   |
    //                         +---------+  ========  | |  +--------+
    //                         | bgImage |->| img  |->|/
    //                         +---------+  ========
    //
    //
    //           _____________________________________________
    //          | two-pass Gaussian blur                      |
    //          |                                             |
    // +----+   |  ==========    +------------+   ==========  |    +----------+
    // | bg |---|->|  blur  |--> | blurFirst  |-->|  blur  |--|--> | styledBg |
    // +----+   |  | Shader |    | PassResult |   | Shader |  |    +----------+
    //          |  ==========    +------------+   ==========  |
    //          |_____________________________________________|
    //

    this.guidedFilter.setup(
        this.guidedFilterRadius_,
        1e-6,
        this.clippedSize_[0] * this.zoom_,
        this.clippedSize_[1] * this.zoom_,
    );

    this.utils.setup2dQuad();

    switch (this.effect_) {
      case 'label': {
        this.setupColorizeShader_();
      } break;
      case 'blur': {
        this.setupExtractShader_();
        this.setupBlurShader_();
        this.setupBlendShader_();
      } break;
      case 'image': {
        this.setupExtractShader_();
        this.setupImageShader_();
        this.setupBlendShader_();
      } break;
      case 'fill': {
        this.setupExtractShader_();
        this.setupFillShader_();
        this.setupBlendShader_();
      } break;
      default: {
        console.warn('Unknown effect');
      }
    }
  }

  setupColorizeShader_() {
    const vs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      out vec2 v_maskcord;

      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
        v_maskcord = v_texcoord;
      }`;

    const fs =
      `#version 300 es
      precision highp float;
      out vec4 out_color;

      uniform sampler2D u_image;
      uniform sampler2D upredictions_;
      uniform sampler2D u_palette;
      uniform int u_length;
      uniform float u_alpha;

      in vec2 v_maskcord;
      in vec2 v_texcoord;

      void main() {
        float label_index = texture(upredictions_, v_maskcord).a * 255.0;
        vec4 label_color = texture(u_palette, vec2((label_index + 0.5) / float(u_length), 0.5));
        vec4 im_color = texture(u_image, v_texcoord);
        out_color = mix(im_color, label_color, u_alpha);
      }`;

    this.shaders.colorize = new Shader(this.gl, vs, fs);
    this.shaders.colorize.use();
    this.shaders.colorize.set1i('u_image', 0); // texture units 0
    this.shaders.colorize.set1i('upredictions_', 1); // texture units 1
    this.shaders.colorize.set1i('u_palette', 2); // texture units 2
    this.shaders.colorize.set1i('u_length', this.colorPalette_.length / 3);

    if (typeof this.utils.getTexture('image') === 'undefined') {
      this.utils.createAndBindTexture({
        name: 'image',
        filter: this.gl.LINEAR,
      });
    }

    this.utils.createAndBindTexture({
      name: 'predictions',
      filter: this.gl.NEAREST,
    });

    this.utils.createAndBindTexture({
      name: 'palette',
      filter: this.gl.NEAREST,
    });

    this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGB,
        this.colorPalette_.length / 3,
        1,
        0,
        this.gl.RGB,
        this.gl.UNSIGNED_BYTE,
        this.colorPalette_,
    );
  }

  setupExtractShader_() {
    // When guided filter pre-processing is disabled, raw mask is used as input
    // However, raw mask texture is slightly biased and horizontally flipped
    const vsWithoutPreprocess =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      out vec2 v_maskcord;

      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
        v_maskcord = v_texcoord;
      }`;

    const vsWithPreprocess =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      out vec2 v_maskcord;

      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
        v_maskcord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
      }`;

    const fs =
      `#version 300 es
      precision highp float;
      layout(location = 0) out vec4 fg_color;
      layout(location = 1) out vec4 bg_color;

      uniform sampler2D u_mask;
      uniform sampler2D u_image;

      in vec2 v_maskcord;
      in vec2 v_texcoord;

      void main() {
        float fg_alpha = texture(u_mask, v_maskcord).a;
        float bg_alpha = 1.0 - fg_alpha;

        vec4 pixel = texture(u_image, v_texcoord);
        fg_color = vec4(pixel.xyz * fg_alpha, fg_alpha);
        bg_color = vec4(pixel.xyz * bg_alpha, bg_alpha);
      }`;

    if (this.guidedFilterRadius_ === 0) {
      // guided filter is disabled
      this.shaders.extract = new Shader(this.gl, vsWithoutPreprocess, fs);
    } else {
      this.shaders.extract = new Shader(this.gl, vsWithPreprocess, fs);
    }

    if (typeof this.utils.getTexture('image') === 'undefined') {
      this.utils.createAndBindTexture({
        name: 'image',
        filter: this.gl.LINEAR,
      });
    }

    this.utils.createAndBindTexture({
      name: 'predictions',
      filter: this.gl.NEAREST,
    });

    this.utils.createTexInFrameBuffer('extract',
        [{
          name: 'fg',
          width: this.clippedSize_[0] * this.zoom_,
          height: this.clippedSize_[1] * this.zoom_,
        }, {
          name: 'bg',
          width: this.clippedSize_[0] * this.zoom_,
          height: this.clippedSize_[1] * this.zoom_,
        }],
    );
    this.shaders.extract.use();
    this.shaders.extract.set1i('u_image', 0); // texture units 0
    this.shaders.extract.set1i('u_mask', 1); // texture units 1
  }

  setupBlurShader_() {
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
      uniform float kernel[${this.blurRadius_ + 1}];

      // https://learnopengl.com/code_viewer_gh.php?code=src/5.advanced_lighting/7.bloom/7.blur.fs

      void main() {
        vec2 tex_offset = 1.0 / vec2(textureSize(bg, 0)); // gets size of single texel
        vec4 bg_color = texture(bg, v_texcoord);
        vec4 result = bg_color * kernel[0];
        if (first_pass) {
          for (int i = 1; i < ${this.blurRadius_ + 1}; ++i) {
            result += texture(bg, v_texcoord + vec2(tex_offset.x * float(i), 0.0)) * kernel[i];
            result += texture(bg, v_texcoord - vec2(tex_offset.x * float(i), 0.0)) * kernel[i];
          }
        } else {
          for (int i = 1; i < ${this.blurRadius_ + 1}; ++i) {
            result += texture(bg, v_texcoord + vec2(0.0, tex_offset.y * float(i))) * kernel[i];
            result += texture(bg, v_texcoord - vec2(0.0, tex_offset.y * float(i))) * kernel[i];
          }
        }
        out_color = result;
      }`;

    this.shaders.blur = new Shader(this.gl, vs, fs);
    this.shaders.blur.use();
    this.shaders.blur.set1fv('kernel', this.halfKernel_);

    this.utils.createTexInFrameBuffer('blurFirstPassResult',
        [{
          name: 'blurFirstPassResult',
          width: this.clippedSize_[0] * this.zoom_,
          height: this.clippedSize_[1] * this.zoom_,
        }],
    );

    this.utils.createTexInFrameBuffer('styledBg',
        [{
          name: 'styledBg',
          width: this.clippedSize_[0] * this.zoom_,
          height: this.clippedSize_[1] * this.zoom_,
        }],
    );
  }

  setupFillShader_() {
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

      in vec2 v_texcoord;
      out vec4 out_color;
      uniform vec4 fill_color;

      void main() {
        // solid color background
        out_color = fill_color;
      }`;

    this.shaders.fill = new Shader(this.gl, vs, fs);
    this.shaders.fill.use();
    // set solid color in fill shader
    const fillColor = this.bgColor_.map((x) => x / 255);
    this.shaders.fill.set4f('fill_color', ...fillColor, 1);

    this.utils.createTexInFrameBuffer('styledBg',
        [{
          name: 'styledBg',
          width: this.clippedSize_[0] * this.zoom_,
          height: this.clippedSize_[1] * this.zoom_,
        }],
    );
  }

  setupImageShader_() {
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

      in vec2 v_texcoord;
      out vec4 out_color;
      uniform int has_bg;
      uniform vec2 canvas_size;

      uniform sampler2D bg_image;

      void main() {
        if (has_bg != 0) {
          // stretch the background image to fit the viewport
          vec2 image_size = vec2(textureSize(bg_image, 0));
          float image_ratio = image_size.x / image_size.y;
          float canvas_ratio = canvas_size.x / canvas_size.y;
          vec2 bgcoord = (v_texcoord / image_size) * canvas_size;
          if (image_ratio > canvas_ratio) {
            bgcoord *= image_size.y / canvas_size.y;
          } else {
            bgcoord *= image_size.x / canvas_size.x;
          }
          out_color = texture(bg_image, bgcoord);
        } else {
          // checkerboard background
          vec2 offset = floor(v_texcoord * canvas_size / 5.0);
          if (mod(offset.x, 2.0) == 0.0) {
            if (mod(offset.y, 2.0) == 0.0) {
              out_color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
              out_color = vec4(0.8, 0.8, 0.8, 1.0);
            }
          } else {
            if (mod(offset.y, 2.0) != 0.0) {
              out_color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
              out_color = vec4(0.8, 0.8, 0.8, 1.0);
            }
          }
        }
      }`;

    this.shaders.image = new Shader(this.gl, vs, fs);
    this.utils.createAndBindTexture({name: 'bgImage'});
    this.utils.createTexInFrameBuffer('styledBg',
        [{
          name: 'styledBg',
          width: this.clippedSize_[0] * this.zoom_,
          height: this.clippedSize_[1] * this.zoom_,
        }],
    );

    this.shaders.image.use();
    this.shaders.image.set2f(
        'canvas_size',
        this.clippedSize_[0],
        this.clippedSize_[1],
    );
    if (this.backgroundImageSource_) {
      this.shaders.image.set1i('has_bg', 1);
      this.utils.bindTexture('bgImage');
      this.gl.texImage2D(
          this.gl.TEXTURE_2D,
          0,
          this.gl.RGBA,
          this.gl.RGBA,
          this.gl.UNSIGNED_BYTE,
          this.backgroundImageSource_,
      );
    } else {
      this.shaders.image.set1i('has_bg', 0);
    }
  }

  setupBlendShader_() {
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
      out vec4 out_color;

      uniform sampler2D fg;
      uniform sampler2D bg;
      uniform sampler2D orig;

      void main() {
        vec4 fg_color = texture(fg, v_texcoord);
        vec4 bg_color = texture(bg, v_texcoord);
        vec4 orig_color = texture(orig, vec2(v_texcoord.x, (1.0-v_texcoord.y)));
        bg_color = bg_color + (1.0 - bg_color.a) * orig_color;
        out_color = fg_color + (1.0 - fg_color.a) * bg_color;
      }`;

    this.shaders.blend = new Shader(this.gl, vs, fs);
    this.shaders.blend.use();
    this.shaders.blend.set1i('fg', 0); // texture units 0
    this.shaders.blend.set1i('bg', 1); // texture units 1
    this.shaders.blend.set1i('orig', 2); // texture units 2
  }

  async uploadNewTexture(imageSource, clippedSize) {
    if (this.clippedSize_[0] !== clippedSize[0] ||
      this.clippedSize_[1] !== clippedSize[1]) {
      this.clippedSize_ = clippedSize;
      this.zoom_ = this.fixedSize_ / this.clippedSize_[0];

      // all FRAMEBUFFERs should be reconfigured when clippedSize changes
      this.setup();
    }

    this.imageSource_ = imageSource;

    this.utils.bindTexture('image');
    this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        this.gl.RGBA,
        this.gl.UNSIGNED_BYTE,
        imageSource,
    );
  }

  async drawOutputs(newSegMap) {
    if (!newSegMap) return;

    this.gl.canvas.width = this.clippedSize_[0] * this.zoom_;
    this.gl.canvas.height = this.clippedSize_[1] * this.zoom_;
    this.utils.setViewport(this.gl.drawingBufferWidth, this.gl.drawingBufferHeight);

    if (this.effect_ === 'label') {
      // Display color labels
      this.segMap_ = newSegMap;
      this.predictions_ = this.maskSegMap_(newSegMap);
      this.utils.bindTexture('predictions');
      this.gl.pixelStorei(this.gl.UNPACK_ALIGNMENT, 1);
      this.gl.texImage2D(
          this.gl.TEXTURE_2D,
          0,
          this.gl.ALPHA,
          this.clippedSize_[0],
          this.clippedSize_[1],
          0,
          this.gl.ALPHA,
          this.gl.UNSIGNED_BYTE,
          this.predictions_,
      );
      this.drawColorLabel_();
    } else {
      // Person segmentation
      this.segMap_ = newSegMap;
      this.predictions_ = this.maskSegMapPerson_(newSegMap);
      if (this.guidedFilterRadius_ === 0) {
        // guided filter is disabled
        this.utils.bindTexture('predictions');
        this.gl.pixelStorei(this.gl.UNPACK_ALIGNMENT, 1);
        this.gl.texImage2D(
            this.gl.TEXTURE_2D,
            0,
            this.gl.ALPHA,
            this.clippedSize_[0],
            this.clippedSize_[1],
            0,
            this.gl.ALPHA,
            this.gl.UNSIGNED_BYTE,
            this.predictions_,
        );
      } else {
        // guided filter is enabled
        const refinedMask = this.guidedFilter.apply(
            this.imageSource_,
            this.predictions_,
            this.clippedSize_[0],
            this.clippedSize_[1],
        );
        this.utils.setTexture('predictions', refinedMask);
      }
      this.drawPerson_();
    }
  }

  drawColorLabel_() {
    const currShader = this.shaders.colorize;
    currShader.use();
    currShader.set1f('u_alpha', this.colorMapAlpha_);
    this.utils.bindFramebuffer(null);
    this.utils.bindInputTextures(['image', 'predictions', 'palette']);
    this.utils.render();

    // generate label map. { labelName: [ labelName, rgbTuple ] }
    const uniqueLabels = new Set(this.predictions_);
    const labelMap = {};
    for (const labelId of uniqueLabels) {
      const labelName = this.segMap_.labels[labelId];
      const rgbTuple = this.colorPalette_.slice(labelId * 3, (labelId + 1) * 3);
      labelMap[labelId] = [labelName, rgbTuple];
    }
    this.showLegends_(labelMap);
    return labelMap;
  }

  showLegends_(labelMap) {
    const shownLabelId = new Set();

    $('.seg-label').each((i, e) => {
      const id = $(e).attr('data-label-id');
      if (!Object.prototype.hasOwnProperty.call(labelMap, id)) {
        $(e).remove();
      } else {
        shownLabelId.add(id);
      }
    });

    for (const id in labelMap) {
      if (!shownLabelId.has(id)) {
        const labelDiv = $(`<div class="col-12 seg-label" data-label-id="${id}"/>`)
            .append($(`<span style="color:rgb(${labelMap[id][1]})">⬤</span>`))
            .append(`${labelMap[id][0]}`);
        $('.labels-wrapper').append(labelDiv);
      }
    }
  }

  highlightHoverLabel(hoverPos, outputCanvas) {
    if (!this.predictions_) return;

    if (!hoverPos) {
      // clear highlight when mouse leaves canvas
      $('.seg-label').removeClass('highlight');
      return;
    }

    const outputW = this.clippedSize_[0];
    const actualZoom = outputCanvas.clientWidth / this.clippedSize_[0];

    const x = Math.floor(hoverPos.x / actualZoom);
    const y = Math.floor(hoverPos.y / actualZoom);
    const labelId = this.predictions_[x + y * outputW];

    $('.seg-label').removeClass('highlight');
    $('.labels-wrapper').find(`[data-label-id="${labelId}"]`).addClass('highlight');
  }

  drawPerson_() {
    // feed image and mask into extract shader
    let currShader;
    this.shaders.extract.use();
    this.utils.bindFramebuffer('extract');
    this.utils.bindInputTextures(['image', 'predictions']);
    this.utils.render();

    switch (this.effect_) {
      case 'blur': {
        // feed extracted background into blur shader
        currShader = this.shaders.blur;
        currShader.use();

        currShader.set1i('first_pass', 1);
        this.utils.bindFramebuffer('blurFirstPassResult');
        this.utils.bindInputTextures(['bg']);
        this.utils.render();

        currShader.set1i('first_pass', 0);
        this.utils.bindFramebuffer('styledBg');
        this.utils.bindInputTextures(['blurFirstPassResult']);
        this.utils.render();
      } break;
      case 'image': {
        // set image in image shader
        currShader = this.shaders.image;
        currShader.use();
        this.utils.bindFramebuffer('styledBg');
        this.utils.bindInputTextures(['bgImage']);
        this.utils.render();
      } break;
      case 'fill': {
        currShader = this.shaders.fill;
        currShader.use();
        this.utils.bindFramebuffer('styledBg');
        this.utils.render();
      }
    }

    // feed into blend shader
    this.shaders.blend.use();
    this.utils.bindFramebuffer(null);
    this.utils.bindInputTextures(['fg', 'styledBg', 'image']);
    this.utils.render();
  }

  maskSegMap_(segmap) {
    const clippedHeight = this.clippedSize_[1];
    const clippedWidth = this.clippedSize_[0];
    const outputWidth = segmap.outputShape[1];
    const data = segmap.data;
    const mask = new Uint8Array(clippedHeight * clippedWidth);
    let i = 0;
    for (let h = 0; h < clippedHeight; h++) {
      for (let w = 0; w < clippedWidth; w++) {
        mask[i++] = data[h * outputWidth + w];
      }
    }
    return mask;
  }

  maskSegMapPerson_(segmap) {
    const PERSON_ID = 15;
    const clippedHeight = this.clippedSize_[1];
    const clippedWidth = this.clippedSize_[0];
    const outputWidth = segmap.outputShape[1];
    const data = segmap.data;
    const mask = new Uint8Array(clippedHeight * clippedWidth);
    let i = 0;
    for (let h = 0; h < clippedHeight; h++) {
      for (let w = 0; w < clippedWidth; w++) {
        mask[i++] = data[h * outputWidth + w] === PERSON_ID ? 255 : 0;
      }
    }
    return mask;
  }

  generateGaussianKernel1D_(kernelSize, sigma = 30) {
    const gaussian = (x, sigma) =>
      1 / (Math.sqrt(2 * Math.PI) * sigma) * Math.exp(-x * x / (2 * sigma * sigma));
    const kernel = [];
    const radius = (kernelSize - 1) / 2;
    for (let x = -radius; x <= radius; x++) {
      kernel.push(gaussian(x, sigma));
    }

    // normalize kernel
    const sum = kernel.reduce((x, y) => x + y);
    return kernel.map((x) => x / sum);
  }

  deleteAll() {
    this.utils.delete();
    this.guidedFilter.utils.delete();
  }
}
