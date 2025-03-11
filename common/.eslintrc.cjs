module.exports = {
  ignorePatterns: ['libs/'],
  globals: {
    'BigInt64Array': 'readonly',
    'BigUint64Array': 'readonly',
    'Float16Array': 'readonly',
    "globalThis": true,
    'tf': 'readonly',
  },
  parser: 'babel-eslint'
};
