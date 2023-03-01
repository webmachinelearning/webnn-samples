export const glsl = String.raw;

/**
 * Create stage program for WebGL pipeline
 * @param {WebGL2RenderingContext} gl, gl object
 * @param {WebGLShader} vertexShader, vertex shader
 * @param {WebGLShader} fragmentShader, fragment shader
 * @param {WebGLBuffer} positionBuffer, buffer of position
 * @param {WebGLBuffer} texCoordBuffer, buffer of tex coord
 */
export function createPiplelineStageProgram(
    gl, vertexShader, fragmentShader, positionBuffer, texCoordBuffer) {
  const program = createProgram(gl, vertexShader, fragmentShader);

  const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(positionAttributeLocation);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

  const texCoordAttributeLocation = gl.getAttribLocation(program, 'a_texCoord');
  gl.enableVertexAttribArray(texCoordAttributeLocation);
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.vertexAttribPointer(texCoordAttributeLocation, 2, gl.FLOAT, false, 0, 0);

  return program;
}

export function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(
        `Could not link WebGL program: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

export function compileShader(gl, shaderType, shaderSource) {
  const shader = gl.createShader(shaderType);
  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`Could not compile shader: ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

export function createTexture(
    gl, internalformat, width, height, minFilter, magFilter) {
  const texture = gl.createTexture()
  minFilter = minFilter === undefined ? gl.NEAREST : minFilter;
  magFilter = magFilter === undefined ? gl.NEAREST : magFilter;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internalformat, width, height);

  return texture;
}

export async function readPixelsAsync(
    gl, x, y, width, height, format, type, dest) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buf);
  gl.bufferData(gl.PIXEL_PACK_BUFFER, dest.byteLength, gl.STREAM_READ);
  gl.readPixels(x, y, width, height, format, type, 0);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

  await getBufferSubDataAsync(gl, gl.PIXEL_PACK_BUFFER, buf, 0, dest);

  gl.deleteBuffer(buf);
  return dest;
}

async function getBufferSubDataAsync(
    gl, target, buffer, srcByteOffset, dstBuffer, dstOffset = undefined, length = undefined) {
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  gl.flush();
  const res = await clientWaitAsync(gl, sync);
  gl.deleteSync(sync);

  if (res !== gl.WAIT_FAILED) {
    gl.bindBuffer(target, buffer);
    gl.getBufferSubData(target, srcByteOffset, dstBuffer, dstOffset, length);
    gl.bindBuffer(target, null);
  }
}

function clientWaitAsync(gl, sync) {
  return new Promise((resolve) => {
    function test() {
      const res = gl.clientWaitSync(sync, 0, 0);
      if (res === gl.WAIT_FAILED) {
        resolve(res);
        return;
      }
      if (res === gl.TIMEOUT_EXPIRED) {
        requestAnimationFrame(test);
        return;
      }
      resolve(res);
    }
    requestAnimationFrame(test);
  });
}
