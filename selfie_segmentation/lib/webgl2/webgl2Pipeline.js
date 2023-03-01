import {compileShader, createTexture, glsl} from '../helpers/webglHelper.js';
import {buildBackgroundBlurStage} from './backgroundBlurStage.js';
import {buildBackgroundImageStage} from './backgroundImageStage.js';
import {buildJointBilateralFilterStage} from './jointBilateralFilterStage.js';
import {buildLoadSegmentationStage} from './loadSegmentationStage.js';
import {buildResizingStage} from './resizingStage.js';

export function buildWebGL2Pipeline(
    sourcePlayback, backgroundImage, backgroundType, inputResolution, canvas, outputBuffer) {
  const vertexShaderSource = glsl`#version 300 es

    in vec2 a_position;
    in vec2 a_texCoord;

    out vec2 v_texCoord;

    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
      v_texCoord = a_texCoord;
    }
  `;

  const {width: frameWidth, height: frameHeight} = sourcePlayback;
  const [segmentationWidth, segmentationHeight] = inputResolution;

  const gl = canvas.getContext('webgl2');

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);

  const vertexArray = gl.createVertexArray();
  gl.bindVertexArray(vertexArray);

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    gl.STATIC_DRAW,
  );

  const texCoordBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    gl.STATIC_DRAW,
  );

  // We don't use texStorage2D here because texImage2D seems faster
  // to upload video texture than texSubImage2D even though the latter
  // is supposed to be the recommended way:
  // https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_texstorage_to_create_textures
  const inputFrameTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, inputFrameTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  // TODO Rename segmentation and person mask to be more specific
  const segmentationTexture = createTexture(
    gl,
    gl.RGBA8,
    segmentationWidth,
    segmentationHeight,
  );
  const personMaskTexture = createTexture(
    gl,
    gl.RGBA8,
    frameWidth,
    frameHeight,
  );

  const resizingStage = buildResizingStage(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    inputResolution,
  );
  const loadSegmentationStage = buildLoadSegmentationStage(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    inputResolution,
    outputBuffer,
    segmentationTexture,
  );
  const jointBilateralFilterStage = buildJointBilateralFilterStage(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    segmentationTexture,
    inputResolution,
    personMaskTexture,
    canvas,
  );
  const backgroundStage =
    backgroundType === 'blur'
      ? buildBackgroundBlurStage(
          gl,
          vertexShader,
          positionBuffer,
          texCoordBuffer,
          personMaskTexture,
          canvas,
        )
      : buildBackgroundImageStage(
          gl,
          positionBuffer,
          texCoordBuffer,
          personMaskTexture,
          backgroundImage,
          canvas,
        );

  const render = async function() {
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputFrameTexture);

    // texImage2D seems faster than texSubImage2D to upload
    // video texture
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      sourcePlayback,
    );

    gl.bindVertexArray(vertexArray);

    resizingStage.render();

    loadSegmentationStage.render();
    jointBilateralFilterStage.render();
    backgroundStage.render();
  };

  const updatePostProcessingConfig = function(postProcessingConfig) {
    jointBilateralFilterStage.updateSigmaSpace(
        postProcessingConfig.jointBilateralFilter.sigmaSpace);
    jointBilateralFilterStage.updateSigmaColor(
        postProcessingConfig.jointBilateralFilter.sigmaColor);

    if (backgroundType === 'image') {
      // const backgroundImageStage = backgroundStage as BackgroundImageStage
      backgroundStage.updateCoverage(postProcessingConfig.coverage);
      backgroundStage.updateLightWrapping(postProcessingConfig.lightWrapping);
      backgroundStage.updateBlendMode(postProcessingConfig.blendMode);
    } else if (backgroundType === 'blur') {
      // const backgroundBlurStage = backgroundStage as BackgroundBlurStage
      backgroundStage.updateCoverage(postProcessingConfig.coverage);
    } else {
      // TODO Handle no background in a separate pipeline path
      // const backgroundImageStage = backgroundStage as BackgroundImageStage
      backgroundStage.updateCoverage([0, 0.9999]);
      backgroundStage.updateLightWrapping(0);
    }
  };

  const cleanUp = function() {
    backgroundStage.cleanUp();
    jointBilateralFilterStage.cleanUp();
    loadSegmentationStage.cleanUp();
    resizingStage.cleanUp();

    gl.deleteTexture(personMaskTexture);
    gl.deleteTexture(segmentationTexture);
    gl.deleteTexture(inputFrameTexture);
    gl.deleteBuffer(texCoordBuffer);
    gl.deleteBuffer(positionBuffer);
    gl.deleteVertexArray(vertexArray);
    gl.deleteShader(vertexShader);
  };

  return {render, updatePostProcessingConfig, cleanUp};
}
