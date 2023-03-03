import {
  compileShader,
  createPiplelineStageProgram,
  createTexture,
  glsl,
} from '../helpers/webglHelper.js';

export function buildLoadSegmentationStage(
    gl, vertexShader, positionBuffer, texCoordBuffer, inputResolution, outputBuffer, outputTexture) {
  const fragmentShaderSource = glsl`#version 300 es

    precision highp float;

    uniform sampler2D u_inputSegmentation;

    in vec2 v_texCoord;

    out vec4 outColor;

    void main() {
      float segmentation = texture(u_inputSegmentation, v_texCoord).r;
      outColor = vec4(vec3(0.0), segmentation);
    }
  `;

  const [segmentationWidth, segmentationHeight] = inputResolution;

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
  const inputLocation = gl.getUniformLocation(program, 'u_inputSegmentation');
  const inputTexture = createTexture(
    gl,
    gl.R32F,
    segmentationWidth,
    segmentationHeight,
  );

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    outputTexture,
    0,
  );

  gl.useProgram(program);
  gl.uniform1i(inputLocation, 1);

  const render = function() {
    gl.viewport(0, 0, segmentationWidth, segmentationHeight);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      segmentationWidth,
      segmentationHeight,
      gl.RED,
      gl.FLOAT,
      outputBuffer,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  const cleanUp = function() {
    gl.deleteFramebuffer(frameBuffer);
    gl.deleteTexture(inputTexture);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  };

  return {render, cleanUp};
}
