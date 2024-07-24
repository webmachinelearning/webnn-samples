import {Harness} from './testharness.js';
import {NNotepad} from './nnotepad.js';

// ============================================================
// Helper for NNotepad-specific tests
// ============================================================

async function test(expr, expected) {
  function assert(message, actual, expected) {
    if (Array.isArray(expected)) {
      if (!Object.is(actual.length, expected.length)) {
        throw new Error(`${message} length, expected: ${
          expected.length}, actual: ${actual.length}`);
      }
      for (let i = 0; i < expected.length; ++i) {
        if (!Object.is(actual[i], expected[i])) {
          throw new Error(`${message}[${i}], expected: ${
            expected[i]}, actual: ${actual[i]}`);
        }
      }
    } else if (!Object.is(actual, expected)) {
      throw new Error(`${message}, expected: ${expected}, actual: ${actual}`);
    }
  }

  try {
    const [builderFunc] = NNotepad.makeBuilderFunction(expr);
    const result = await NNotepad.execBuilderFunction('cpu', builderFunc);
    if (!Array.isArray(expected)) {
      assert('single tensor', result.length, 1);
      assert('dataType', result[0].dataType, expected.dataType);
      assert('shape', result[0].shape, expected.shape);
      assert('buffer', [...result[0].buffer], expected.buffer);
    } else {
      assert('number of outputs', result.length, expected.length);
      for (let i = 0; i < expected.length; ++i) {
        assert('dataType', result[i].dataType, expected[i].dataType);
        assert('shape', result[i].shape, expected[i].shape);
        assert('buffer', [...result[i].buffer], expected[i].buffer);
      }
    }
    Harness.ok(`ok: ${expr}`);
  } catch (ex) {
    Harness.error(`failed: ${expr} - ${ex.message}`);
  }
}

async function testThrows(expr) {
  try {
    const [builderFunc] = NNotepad.makeBuilderFunction(expr);
    await NNotepad.execBuilderFunction('cpu', builderFunc);
    Harness.error(`failed: ${expr} - expected to throw`);
  } catch (ex) {
    Harness.ok(`ok: ${expr}`);
  }
}

// ============================================================
// Test Cases
// ============================================================

document.addEventListener('DOMContentLoaded', async (e) => {
  Harness.section('Numbers');
  await test('125', {dataType: 'float32', shape: [], buffer: [125]});
  await test('-125', {dataType: 'float32', shape: [], buffer: [-125]});
  await test('1.25', {dataType: 'float32', shape: [], buffer: [1.25]});
  await test('1.25e2', {dataType: 'float32', shape: [], buffer: [125]});
  await test('125e-2', {dataType: 'float32', shape: [], buffer: [1.25]});
  await test('Infinity', {dataType: 'float32', shape: [], buffer: [Infinity]});
  await test(
      '-Infinity', {dataType: 'float32', shape: [], buffer: [-Infinity]});
  await test('NaN', {dataType: 'float32', shape: [], buffer: [NaN]});

  Harness.section('Operators');
  await test('1 + 2', {dataType: 'float32', shape: [], buffer: [3]});
  await test('2 * 3', {dataType: 'float32', shape: [], buffer: [6]});
  await test('3 / 2', {dataType: 'float32', shape: [], buffer: [1.5]});
  await test('2 ^ 3', {dataType: 'float32', shape: [], buffer: [8]});
  await test('-(1)', {dataType: 'float32', shape: [], buffer: [-1]});
  await test('--(1)', {dataType: 'float32', shape: [], buffer: [1]});

  await test('1 < 2', {dataType: 'uint8', shape: [], buffer: [1]});
  await test('2 < 1', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('1 < 1', {dataType: 'uint8', shape: [], buffer: [0]});

  await test('1 <= 2', {dataType: 'uint8', shape: [], buffer: [1]});
  await test('2 <= 1', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('1 <= 1', {dataType: 'uint8', shape: [], buffer: [1]});

  await test('1 > 2', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('2 > 1', {dataType: 'uint8', shape: [], buffer: [1]});
  await test('1 > 1', {dataType: 'uint8', shape: [], buffer: [0]});

  await test('1 >= 2', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('2 >= 1', {dataType: 'uint8', shape: [], buffer: [1]});
  await test('1 >= 1', {dataType: 'uint8', shape: [], buffer: [1]});

  await test('1 == 2', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('2 == 0', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('1 == 1', {dataType: 'uint8', shape: [], buffer: [1]});

  await test('!1u8', {dataType: 'uint8', shape: [], buffer: [0]});
  await test('!0u8', {dataType: 'uint8', shape: [], buffer: [1]});
  await test('!!1u8', {dataType: 'uint8', shape: [], buffer: [1]});
  await test('!!0u8', {dataType: 'uint8', shape: [], buffer: [0]});

  Harness.section('Scalar type suffixes');
  await test('-123i8', {dataType: 'int8', shape: [], buffer: [-123]});
  await test('123u8', {dataType: 'uint8', shape: [], buffer: [123]});
  await test('-123i32', {dataType: 'int32', shape: [], buffer: [-123]});
  await test('123u32', {dataType: 'uint32', shape: [], buffer: [123]});
  await test('-123i64', {dataType: 'int64', shape: [], buffer: [-123n]});
  await test('123u64', {dataType: 'uint64', shape: [], buffer: [123n]});
  await test(
      '12.34f32',
      {dataType: 'float32', shape: [], buffer: [Math.fround(12.34)]});
  await test('12.34f16', {dataType: 'float16', shape: [], buffer: [12.34375]});

  Harness.section('Tensor type suffixes');
  await test('[-123]i8', {dataType: 'int8', shape: [1], buffer: [-123]});
  await test('[123]u8', {dataType: 'uint8', shape: [1], buffer: [123]});
  await test('[-123]i32', {dataType: 'int32', shape: [1], buffer: [-123]});
  await test('[123]u32', {dataType: 'uint32', shape: [1], buffer: [123]});
  await test('[-123]i64', {dataType: 'int64', shape: [1], buffer: [-123n]});
  await test('[123]u64', {dataType: 'uint64', shape: [1], buffer: [123n]});
  await test(
      '[12.34]f32',
      {dataType: 'float32', shape: [1], buffer: [Math.fround(12.34)]});
  await test(
      '[12.34]f16', {dataType: 'float16', shape: [1], buffer: [12.34375]});

  Harness.section('Tensors');
  await test(
      `A = [[1,7],[2,4]]  B = [[3,3],[5,2]]  matmul(A,B)`,
      {dataType: 'float32', shape: [2, 2], buffer: [38, 17, 26, 14]});
  await test(
      `M = [[2,8,3],[5,4,1]]  N = [[4,1],[6,3],[2,4]]  matmul(M,N)`,
      {dataType: 'float32', shape: [2, 2], buffer: [62, 38, 46, 21]});

  Harness.section('Dictionaries');
  await test('linear(10, {})', {dataType: 'float32', shape: [], buffer: [10]});
  await test(
      'linear(10, {alpha: 2, beta: 3})',
      {dataType: 'float32', shape: [], buffer: [23]});

  Harness.section('String arguments');
  await test(
      `cast([1,2,3], 'int8')`,
      {dataType: 'int8', shape: [3], buffer: [1, 2, 3]});
  await test(
      `cast([1,2,3], "int8")`,
      {dataType: 'int8', shape: [3], buffer: [1, 2, 3]});

  Harness.section('Multiple output tensors');
  await test(`split([1,2,3,4], 2)`, [
    {dataType: 'float32', shape: [2], buffer: [1, 2]},
    {dataType: 'float32', shape: [2], buffer: [3, 4]},
  ]);

  Harness.section('Non-operand arguments: array of operands');
  await test(
      `A = [1,2]  B = [3,4]  concat([A,B], 0)`,
      {dataType: 'float32', shape: [4], buffer: [1, 2, 3, 4]});
  await test(
      `concat([identity([1,2]),identity([3,4])], 0)`,
      {dataType: 'float32', shape: [4], buffer: [1, 2, 3, 4]});

  Harness.section('Non-operand arguments: array of numbers');
  await test(
      `T = [[1,2,3],[4,5,6]]  reshape(T, [1, 3, 2, 1])`,
      {dataType: 'float32', shape: [1, 3, 2, 1], buffer: [1, 2, 3, 4, 5, 6]});
  await test(
      `expand([1], [2, 2])`,
      {dataType: 'float32', shape: [2, 2], buffer: [1, 1, 1, 1]});

  Harness.section('Non-operand arguments: simple numbers');
  await test(
      `softmax([1], 0)`,
      {dataType: 'float32', shape: [1], buffer: [1]});

  Harness.section('Regression tests');
  await test(
      `concat([[1,2],[3,4]], 0)`,
      {dataType: 'float32', shape: [4], buffer: [1, 2, 3, 4]});
  await test(
      `trueblue = 123  (trueblue) + 1`,
      {dataType: 'float32', shape: [], buffer: [124]});
  await test(
      `InfinityGauntlet = 123  (InfinityGauntlet) + 1`,
      {dataType: 'float32', shape: [], buffer: [124]});
  await test(
      `NaNBread = 123  (NaNBread) + 1`,
      {dataType: 'float32', shape: [], buffer: [124]});
  await testThrows(`123u88`);
  // await test(`input = [[[1,2],[3,4]],[[5,6],[7,8]]]  weight =
  // [[[1,2],[1,2],[1,2],[1,2]]]  rweight = [[[1],[1],[1],[1]]]  lstm(input,
  // weight, rweight, 2, 1)`, {});
});
