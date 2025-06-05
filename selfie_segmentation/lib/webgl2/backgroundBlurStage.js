import {
  compileShader,
  createPiplelineStageProgram,
  createTexture,
  glsl,
} from '../helpers/webglHelper.js';

export function buildBackgroundBlurStage(
    gl, vertexShader, positionBuffer, texCoordBuffer, personMaskTexture, canvas) {
  const blurPass = buildBlurPass(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    personMaskTexture,
    canvas,
  );
  const blendPass = buildBlendPass(gl, positionBuffer, texCoordBuffer, canvas);

  const render = function() {
    blurPass.render();
    blendPass.render();
  };

  const updateCoverage = function(coverage) {
    blendPass.updateCoverage(coverage);
  };

  const cleanUp = function() {
    blendPass.cleanUp();
    blurPass.cleanUp();
  };

  return {
    render,
    updateCoverage,
    cleanUp,
  };
}

function buildBlurPass(
    gl, vertexShader, positionBuffer, texCoordBuffer, personMaskTexture, canvas) {
  const fragmentShaderSource = glsl`#version 300 es

    precision highp float;

    uniform sampler2D u_inputFrame;
    uniform sampler2D u_personMask;
    uniform vec2 u_texelSize;

    in vec2 v_texCoord;

    out vec4 outColor;

    const int radius = 50;

    void main() {
      vec4 frameColor = vec4(0.0);
      float totalWeight = 0.0;

      for (int i = -radius; i <= radius; i++) {
        float weight = exp(-float(i * i) / float(radius * radius));
        vec2 offset = vec2(float(i)) * u_texelSize;

        vec4 sampleColor = texture(u_inputFrame, v_texCoord + offset);
        float sampleMask = texture(u_personMask, v_texCoord + offset).a;

        frameColor += sampleColor * weight * (1.0 - sampleMask);
        totalWeight += weight * (1.0 - sampleMask);
      }

      frameColor /= totalWeight;
      outColor = mix(texture(u_inputFrame, v_texCoord), frameColor, 1.0 - texture(u_personMask, v_texCoord).a);
    }
  `;
  const scale = 0.5;
  const outputWidth = canvas.width * scale;
  const outputHeight = canvas.height * scale;
  const texelWidth = 1 / outputWidth;
  const texelHeight = 1 / outputHeight;

  const fragmentShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource,
  );
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer,
  );
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const personMaskLocation = gl.getUniformLocation(program, 'u_personMask');
  const texelSizeLocation = gl.getUniformLocation(program, 'u_texelSize');
  const texture1 = createTexture(
    gl,
    gl.RGBA8,
    outputWidth,
    outputHeight,
    gl.NEAREST,
    gl.LINEAR,
  );
  const texture2 = createTexture(
    gl,
    gl.RGBA8,
    outputWidth,
    outputHeight,
    gl.NEAREST,
    gl.LINEAR,
  );

  const frameBuffer1 = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer1);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    texture1,
    0,
  );

  const frameBuffer2 = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer2);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    texture2,
    0,
  );

  gl.useProgram(program);
  gl.uniform1i(personMaskLocation, 1);

  const render = function() {
    gl.viewport(0, 0, outputWidth, outputHeight)
    gl.useProgram(program)
    gl.uniform1i(inputFrameLocation, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, personMaskTexture)

    for (let i = 0; i < 3; i++) {
      gl.uniform2f(texelSizeLocation, 0, texelHeight)
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer1)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_2D, texture1)
      gl.uniform1i(inputFrameLocation, 2)

      gl.uniform2f(texelSizeLocation, texelWidth, 0)
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer2)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      gl.bindTexture(gl.TEXTURE_2D, texture2)
    }
  }

  const cleanUp = function() {
    gl.deleteFramebuffer(frameBuffer2)
    gl.deleteFramebuffer(frameBuffer1)
    gl.deleteTexture(texture2)
    gl.deleteTexture(texture1)
    gl.deleteProgram(program)
    gl.deleteShader(fragmentShader)
  }

  return {
    render,
    cleanUp,
  };
}

function buildBlendPass(gl, positionBuffer, texCoordBuffer, canvas) {
  const vertexShaderSource = glsl`#version 300 es

    in vec2 a_position;
    in vec2 a_texCoord;

    out vec2 v_texCoord;

    void main() {
      // Flipping Y is required when rendering to canvas
      gl_Position = vec4(a_position * vec2(1.0, -1.0), 0.0, 1.0);
      v_texCoord = a_texCoord;
    }
  `;

  const fragmentShaderSource = glsl`#version 300 es

    precision highp float;

    uniform sampler2D u_inputFrame;
    uniform sampler2D u_personMask;
    uniform sampler2D u_blurredInputFrame;
    uniform vec2 u_coverage;

    in vec2 v_texCoord;

    out vec4 outColor;

    void main() {
      vec3 color = texture(u_inputFrame, v_texCoord).rgb;
      vec3 blurredColor = texture(u_blurredInputFrame, v_texCoord).rgb;
      float personMask = texture(u_personMask, v_texCoord).a;
      personMask = smoothstep(u_coverage.x, u_coverage.y, personMask);
      outColor = vec4(mix(blurredColor, color, personMask), 1.0);
    }
  `;

  const { width: outputWidth, height: outputHeight } = canvas;

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource,
  );
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer,
  );
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const personMaskLocation = gl.getUniformLocation(program, 'u_personMask');
  const blurredInputFrame = gl.getUniformLocation(
    program,
    'u_blurredInputFrame',
  );
  const coverageLocation = gl.getUniformLocation(program, 'u_coverage');

  gl.useProgram(program);
  gl.uniform1i(inputFrameLocation, 0);
  gl.uniform1i(personMaskLocation, 1);
  gl.uniform1i(blurredInputFrame, 2);
  gl.uniform2f(coverageLocation, 0, 1);

  const render = function() {
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  const updateCoverage = function(coverage) {
    gl.useProgram(program);
    gl.uniform2f(coverageLocation, coverage[0], coverage[1]);
  }

  const cleanUp = function() {
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
    gl.deleteShader(vertexShader);
  }

  return {
    render,
    updateCoverage,
    cleanUp,
  };
}
