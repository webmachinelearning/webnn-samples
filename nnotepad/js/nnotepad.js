/* global BigInt64Array, BigUint64Array, Float16Array */

import {Util} from './util.js';

// ============================================================
// General Utilities
// ============================================================

export class ParseError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ParseError';
  }
}

export class BuildError extends Error {
  constructor(message) {
    super(message);
    this.name = 'build()';
  }
}

export class DispatchError extends Error {
  constructor(message) {
    super(message);
    this.name = 'dispatch()';
  }
}

// ============================================================
// General WebNN Utilities
// ============================================================

const kArgTypeOperandList = 1;
const kArgTypeNonOperand = 2;
const kArgTypeOperand = 3;

class WebNNUtil {
  static bufferForOperand(operand) {
    const size = [...operand.shape()].reduce((a, b) => a * b, 1);
    const ctor = WebNNUtil.dataTypeToBufferType(operand.dataType());
    return Reflect.construct(ctor, [size]);
  }

  static async tensorForOperand(operand, context) {
    const desc = {
      dataType: operand.dataType(),
      dimensions: operand.shape(),
      shape: operand.shape(),
      usage: MLTensorUsage.READ,
    };
    const tensor = await context.createTensor(desc);
    return tensor;
  }

  static dataTypeToBufferType(type) {
    switch (type) {
      case 'int8':
        return Int8Array;
      case 'uint8':
        return Uint8Array;
      case 'int32':
        return Int32Array;
      case 'uint32':
        return Uint32Array;
      case 'int64':
        return BigInt64Array;
      case 'uint64':
        return BigUint64Array;
      case 'float16':
        return Float16Array;
      case 'float32':
        return Float32Array;
    }
    throw new Error(`Unsupported dataType ${type}`);
  }

  static argumentType(name, index) {
    return ({
      concat: {0: kArgTypeOperandList, 1: kArgTypeNonOperand},
      expand: {1: kArgTypeNonOperand},
      gru: {3: kArgTypeNonOperand, 4: kArgTypeNonOperand},
      gruCell: {4: kArgTypeNonOperand},
      lstm: {3: kArgTypeNonOperand, 4: kArgTypeNonOperand},
      lstmCell: {5: kArgTypeNonOperand},
      pad: {1: kArgTypeNonOperand, 2: kArgTypeNonOperand},
      reshape: {1: kArgTypeNonOperand},
      slice: {1: kArgTypeNonOperand, 2: kArgTypeNonOperand},
      softmax: {1: kArgTypeNonOperand},
      split: {1: kArgTypeNonOperand},
    })[name]
        ?.[index] ||
        kArgTypeOperand;
  }
}

export class NNotepad {
  // ============================================================
  // Script Converter
  // ============================================================

  // Returns a tuple:
  // * async JS function with input MLGraphBuilder and output MLOperand
  // * the source to the body of the function

  static makeBuilderFunction(text) {
    // Operators
    const kAdditiveOperators = {
      '+': 'add',
      '-': 'sub',
    };
    const kMultiplicativeOperators = {
      '*': 'mul',
      '/': 'div',
    };
    const kPowerOperators = {
      '^': 'pow',
    };
    const kRelationalOperators = {
      '==': 'equal',
      '>': 'greater',
      '>=': 'greaterOrEqual',
      '<': 'lesser',
      '<=': 'lesserOrEqual',
    };
    const kBinaryOperators = Object.assign(
        {}, kAdditiveOperators, kMultiplicativeOperators, kPowerOperators,
        kRelationalOperators);

    const kUnaryOperators = {
      '-': 'neg',
      '!': 'logicalNot',
    };

    const kDefaultDataType = 'float32';

    // ------------------------------------------------------------
    // Tokenizer

    // Output `tokens` is an array; each token is one of:
    // * a comment
    // * a number
    // * a string
    // * a boolean
    // * a type suffix
    // * an identifier
    // * an operator (single character or digraph)
    // e.g. 'sqrt(a + 12)' -> ['sqrt', '(', 'a', '+', 'b', '12', ')']

    const kOperators = Object.assign({}, kBinaryOperators, kUnaryOperators);

    // Tokens
    const kCommentPattern = '(#|//).*';
    const kNumberPattern =
        'NaN\\b|Infinity\\b|-Infinity\\b|-?\\d+(\\.\\d+)?([eE]-?\\d+)?';
    const kStringPattern =
        `"([^\\\\\\x0A\\x0D"]|\\\\.)*"|'([^\\\\\\x0A\\x0D']|\\\\.)*'`;
    const kBooleanPattern = 'true\\b|false\\b';
    const kSuffixPattern =
        `u8\\b|u32\\b|u64\\b|i8\\b|i32\\b|i64\\b|f16\\b|f32\\b`;
    const kIdentifierPattern = '[A-Za-z]\\w*';

    const rescape = (s) => s.replace(/[\\^$*+?.()|[\]{}]/g, '\\$&');
    const longestFirst = (a, b) => b.length - a.length;
    const kTokenRegEx = new RegExp(
        [
          kCommentPattern,
          kNumberPattern,
          kStringPattern,
          kBooleanPattern,
          kSuffixPattern,
          kIdentifierPattern,
          ...Object.keys(kOperators).sort(longestFirst).map(rescape),
          '.',
        ].join('|'),
        'g');

    const toRegEx = (p) => new RegExp('^(' + p + ')$');
    const kCommentRegEx = toRegEx(kCommentPattern);
    const kNumberRegEx = toRegEx(kNumberPattern);
    const kStringRegEx = toRegEx(kStringPattern);
    const kBooleanRegEx = toRegEx(kBooleanPattern);
    const kSuffixRegEx = toRegEx(kSuffixPattern);
    const kIdentifierRegEx = toRegEx(kIdentifierPattern);
    const isComment = (token) => token && token.match(kCommentRegEx);
    const isNumber = (token) => token && token.match(kNumberRegEx);
    const isString = (token) => token && token.match(kStringRegEx);
    const isBoolean = (token) => token && token.match(kBooleanRegEx);
    const isSuffix = (token) => token && token.match(kSuffixRegEx);
    const isIdentifier = (token) => token && token.match(kIdentifierRegEx);

    const tokens = text.match(kTokenRegEx)
        .map((s) => s.trim())
        .filter((s) => s.length && !isComment(s));
    // console.log('tokens: ', tokens);

    // ------------------------------------------------------------
    // Parser

    // Recursive descent parser; see README.md for grammar
    // `lines` is populated with AST

    const lines = [];
    while (tokens.length) {
      lines.push(parseLine());
    }

    function peek(n) {
      n = n || 0;
      return tokens[n];
    }
    function take() {
      return tokens.shift();
    }
    function expect(expected) {
      const token = take();
      if (token !== expected) {
        throw new ParseError(`Expected '${expected}', saw '${token}'`);
      }
    }
    function parseLine() {
      if (isIdentifier(peek()) && peek(1) === '=') {
        const identifier = take();
        take();
        return {type: 'assignment', identifier, expr: parseExpr()};
      }

      return {type: 'expression', expr: parseExpr()};
    }
    function parseExpr() {
      return parseRelExpr();
    }
    function parseRelExpr() {
      let lhs = parseAddExpr();
      while (Object.keys(kRelationalOperators).includes(peek())) {
        const op = take();
        const rhs = parseAddExpr();
        lhs = {lhs, op, rhs};
      }
      return lhs;
    }
    function parseAddExpr() {
      let lhs = parseMulExpr();
      while (Object.keys(kAdditiveOperators).includes(peek())) {
        const op = take();
        const rhs = parseMulExpr();
        lhs = {lhs, op, rhs};
      }
      return lhs;
    }
    function parseMulExpr() {
      let lhs = parsePowExpr();
      while (Object.keys(kMultiplicativeOperators).includes(peek())) {
        const op = take();
        const rhs = parsePowExpr();
        lhs = {lhs, op, rhs};
      }
      return lhs;
    }
    function parsePowExpr() {
      let lhs = parseUnaryExpr();
      while (Object.keys(kPowerOperators).includes(peek())) {
        const op = take();
        const rhs = parseUnaryExpr();
        lhs = {lhs, op, rhs};
      }
      return lhs;
    }
    function parseUnaryExpr() {
      if (Object.keys(kUnaryOperators).includes(peek())) {
        const op = take();
        const rhs = parseUnaryExpr();
        return {op, rhs};
      }
      return parseFinalExpr();
    }
    function parseFinalExpr() {
      const token = take();
      if (isNumber(token)) {
        let dataType = kDefaultDataType;
        if (isSuffix(peek())) {
          dataType = suffixToDataType(take());
        }
        return {type: 'number', value: Number(token), dataType};
      }
      if (isString(token)) {
        return {type: 'string', value: eval(token)};
      }
      if (isBoolean(token)) {
        return {type: 'boolean', value: token === 'true'};
      }
      if (token === '[') {
        const value = parseArray();
        let dataType = kDefaultDataType;
        if (isSuffix(peek())) {
          dataType = suffixToDataType(take());
        }
        return {type: 'array', value, dataType};
      }
      if (token === '{') {
        const dict = parseDict();
        return {type: 'dict', dict};
      }
      if (isIdentifier(token)) {
        if (peek() !== '(') {
          return {type: 'identifier', value: token};
        }
        take();
        const args = [];
        if (peek() !== ')') {
          args.push(parseExpr());
          while (peek() === ',') {
            take();
            args.push(parseExpr());
          }
        }
        expect(')');
        return {type: 'call', identifier: token, args};
      }
      if (token === '(') {
        const expr = parseExpr();
        expect(')');
        return expr;
      }
      throw new ParseError(`Expected expression, saw '${token}'`);
    }
    function parseArray() {
      const array = [];
      if (peek() !== ']') {
        const expr = parseExpr();
        array.push(expr);
        while (peek() === ',') {
          take();
          const expr = parseExpr();
          array.push(expr);
        }
      }
      expect(']');
      return array;
    }
    function parseDict() {
      const dict = {};
      if (isIdentifier(peek()) || isString(peek())) {
        const [key, value] = parsePropDef();
        dict[key] = value;
        while (peek() === ',') {
          take();
          if (peek() === '}') {
            break;
          }
          if (!(isIdentifier(peek()) || isString(peek()))) {
            throw new ParseError(`Expected identifier, saw '${peek()}'`);
          }
          const [key, value] = parsePropDef();
          dict[key] = value;
        }
      }
      expect('}');
      return dict;

      function parsePropDef() {
        let key = take();
        if (isString(key)) {
          key = eval(key);
        }
        expect(':');
        const expr = parseExpr();
        return [key, expr];
      }
    }

    // ------------------------------------------------------------
    // Serializer

    // Generates WebNN code as the body of a function. `_` is passed as the
    // `MLGraphBuilder`. The output of the last expression is returned.

    const src = lines
        .map(
            (line, index) =>
              serializeLine(line, index === lines.length - 1))
        .map((line) => line + ';\n')
        .join('');
    const AsyncFunction = async function() {}.constructor;
    return [new AsyncFunction(['_', 'Util'], src), src];

    function serializeLine(line, last) {
      const expr = serializeExpr(line.expr);
      switch (line.type) {
        case 'assignment':
          return last ? `return ${expr}` : `const ${line.identifier} = ${expr}`;
        case 'expression':
          return last ? `return ${expr}` : expr;
      }
      throw new Error(`unexpected line type: ${line.type}`);
    }
    function serializeExpr(expr, argumentType = kArgTypeOperand) {
      if (expr.op) {
        if (expr.lhs) {
          return `_.${kBinaryOperators[expr.op]}(${serializeExpr(expr.lhs)}, ${
            serializeExpr(expr.rhs)})`;
        } else {
          return `_.${kUnaryOperators[expr.op]}(${serializeExpr(expr.rhs)})`;
        }
      }
      switch (expr.type) {
        case 'string':
          return Util.stringify(expr.value);
        case 'boolean':
          return String(expr.value);
        case 'number':
          switch (argumentType) {
            case kArgTypeNonOperand:
              return Util.stringify(expr.value);
            default:
              return serializeScalar(expr.value, expr.dataType);
          }
        case 'array':
          switch (argumentType) {
            case kArgTypeNonOperand:
              return serializeArray(expr.value, kArgTypeNonOperand);
            case kArgTypeOperandList:
              return serializeArray(expr.value, kArgTypeOperand);
            default:
              return serializeTensor(expr.value, expr.dataType);
          }
        case 'dict':
          return serializeDict(expr.dict);
        case 'identifier':
          return expr.value;
        case 'call':
          return serializeCall(expr.identifier, expr.args);
      }
      throw new Error(`unexpected expr type: ${expr.type}`);
    }
    function serializeDict(dict) {
      return '{' +
          Object.keys(dict)
              .map((k) => {
                const v = dict[k];
                k = Util.stringify(k);
                return `${k}: ${serializeExpr(v, kArgTypeNonOperand)}`;
              })
              .join(', ') +
          '}';
    }

    function serializeScalar(number, dataType) {
      const ctor = WebNNUtil.dataTypeToBufferType(dataType);
      // building a 0-D scalar input with empty shape
      return `_.constant({dataType:"${dataType}", dimensions: [], shape: []},
      new ${ctor.name}([${Util.stringifyNumber(number, dataType)}]))`;
    }
    function suffixToDataType(suffix) {
      return {
        'i8': 'int8',
        'u8': 'uint8',
        'i32': 'int32',
        'u32': 'uint32',
        'i64': 'int64',
        'u64': 'uint64',
        'f16': 'float16',
        'f32': 'float32',
      }[suffix];
    }

    function serializeTensor(tensor, dataType) {
      const shape = [];
      const elements = [];
      (function measure(t, d) {
        if (d >= shape.length) {
          shape[d] = t.length;
        } else if (shape[d] !== t.length) {
          throw new Error('Invalid tensor: inconsistent shape');
        }
        t.forEach((e) => {
          if (e.type === 'array') {
            measure(e.value, d + 1);
          } else if (e.type !== 'number') {
            throw new Error(`Invalid tensor: saw ${e.type}`);
          } else if (d + 1 !== shape.length) {
            throw new Error('Invalid tensor: saw scalar');
          } else {
            elements.push(e.value);
          }
        });
      }(tensor, 0));
      const ctor = WebNNUtil.dataTypeToBufferType(dataType);
      return `_.constant({dataType: "${dataType}", dimensions: ${
        Util.stringify(shape)}, shape: ${
        Util.stringify(shape)}}, new ${ctor.name}([${
        elements.map((n) => Util.stringifyNumber(n, dataType)).join(',')}]))`;
    }

    function serializeArray(array, argumentType) {
      return '[' +
          array.map((expr) => serializeExpr(expr, argumentType)).join(', ') +
          ']';
    }

    function serializeCall(name, args) {
      if (name === 'load') {
        const [url, shape, dataType] = args;
        if (url.type !== 'string') {
          throw new TypeError('load(): expected string');
        }
        if (shape.type !== 'array') {
          throw new TypeError('load(): expected array');
        }
        if (dataType.type !== 'string') {
          throw new TypeError('load(): expected string');
        }
        const dims = shape.value.map((expr) => expr.value);
        const ctor = WebNNUtil.dataTypeToBufferType(dataType.value);
        return `_.constant({dataType: "${dataType.value}", dimensions: ${
          Util.stringify(dims)}, shape: ${
          Util.stringify(dims)}}, new ${
          ctor.name}(await Util.loadBuffer(${Util.stringify(url.value)})))`;
      }

      if (name === 'zeros') {
        const [shape, dataType] = args;
        if (shape.type !== 'array') {
          throw new TypeError('zeros(): expected array');
        }
        if (dataType.type !== 'string') {
          throw new TypeError('zeros(): expected string');
        }
        const dims = shape.value.map((expr) => expr.value);
        const ctor = WebNNUtil.dataTypeToBufferType(dataType.value);
        const len = dims.reduce((a, b) => a * b, 1);
        return `_.constant({dataType: "${dataType.value}", dimensions: ${
          Util.stringify(dims)}, shape: ${
          Util.stringify(dims)}}, new ${
          ctor.name}(${len}))`;
      }

      return `_.${name}(${
        args.map(
            (arg, index) =>
              serializeExpr(arg, WebNNUtil.argumentType(name, index)))
            .join(', ')})`;
    }
  }

  // ============================================================
  // Script Executor
  // ============================================================

  // Call with the output of `makeBuilderFunc()`. Builds an MLContext and
  // MLGraphBuilder, executes the function to make an MLGraph, then runs
  // dispatch() on it. The output is mapped.

  static async execBuilderFunction(deviceType, builderFunc) {
    const context = await navigator.ml.createContext({deviceType});
    const builder = new self.MLGraphBuilder(context);

    const outputOperands = [];
    let output = await builderFunc(builder, Util);
    if (output instanceof self.MLOperand) {
      // TODO: remove try/catch once all back-ends support `identity()`.
      try {
        // In case `output` is a constant.
        output = builder.identity(output);
      } catch (ex) {
        // Just live with it for now.
      }
      outputOperands.push(output);
    } else if (Array.isArray(output)) {
      outputOperands.push(...output);
      // no-op
    } else {
      throw new ParseError(`Non-MLOperand output: ${output}`);
    }

    const namedOutputs = {};
    const outputTensors = {};
    const outputBuffers = {};
    outputOperands.forEach(async (op, index) => {
      const name = `output-${index}`;
      namedOutputs[name] = op;
      outputBuffers[name] = WebNNUtil.bufferForOperand(op);
      outputTensors[name] = await WebNNUtil.tensorForOperand(op, context);
    });
    let graph;
    try {
      graph = await builder.build(namedOutputs);
    } catch (ex) {
      console.warn(ex);
      throw new BuildError(`${ex.name} : ${ex.message}`);
    }
    const inputTensors = {};

    try {
      context.dispatch(graph, inputTensors, outputTensors);
    } catch (ex) {
      console.warn(ex);
      throw new DispatchError(`${ex.name} : ${ex.message}`);
    }

    for (const [name, outputBuffer] of Object.entries(outputBuffers)) {
      const buffer = await context.readTensor(outputTensors[name]);
      const instance = new outputBuffer.constructor(buffer);
      outputBuffer.set(instance);
    }

    function maybeProxyForFloat16Array(array) {
      return ('proxyForFloat16Array' in self) ?
          self.proxyForFloat16Array(array) :
          array;
    }

    return outputOperands.map(
        (op, index) => ({
          dataType: op.dataType(),
          dimensions: op.shape(),
          shape: op.shape(),
          buffer: maybeProxyForFloat16Array(outputBuffers[`output-${index}`]),
        }));
  }

  // ============================================================
  // Monarch Tokens Provider
  // ============================================================

  // The language ID configured when calling `addMonacoLanguage()`, which
  // should be passed to `monaco.editor.create()`.
  static get monacoLanguageId() {
    return 'nnotepad';
  }

  // Register and configure the NNotepad language with Monaco.

  static addMonacoLanguage(monaco) {
    monaco.languages.register({id: NNotepad.monacoLanguageId});

    monaco.languages.setLanguageConfiguration(
        NNotepad.monacoLanguageId, NNotepad.monacoLanguageConfiguration);

    monaco.languages.setMonarchTokensProvider(
        NNotepad.monacoLanguageId, NNotepad.monarchTokensProvider);

    if ('MLGraphBuilder' in self) {
      // Introspect MLGraphBuilder methods to populate autocompletion.
      const proto = self.MLGraphBuilder.prototype;
      const methods =
          Object.getOwnPropertyNames(proto)
              .map((name) => Object.getOwnPropertyDescriptor(proto, name))
              .filter(
                  (desc) => desc.enumerable && typeof desc.value === 'function')
              .map((desc) => desc.value.name);

      monaco.languages.registerCompletionItemProvider(
          NNotepad.monacoLanguageId, {
            provideCompletionItems: (model, position) => {
              const suggestions = methods.map(
                  (name) => ({
                    label: name,
                    kind: monaco.languages.CompletionItemKind.Keyword,
                    insertText: name,
                  }));
              return {suggestions};
            },
          });
    }
  }

  // Return a Monaco language configuration.
  // https://code.visualstudio.com/api/language-extensions/language-configuration-guide

  static get monacoLanguageConfiguration() {
    return {
      // For comment toggling.
      comments: {
        lineComment: '#',
      },

      // For matching/highlighting.
      brackets: [['{', '}'], ['[', ']'], ['(', ')']],

      // To auto-close as you type the open character.
      autoClosingPairs: [
        {'open': '{', 'close': '}'},
        {'open': '[', 'close': ']'},
        {'open': '(', 'close': ')'},
        {'open': '\'', 'close': '\'', 'notIn': ['string', 'comment']},
        {'open': '"', 'close': '"', 'notIn': ['string']},
      ],
    };
  }

  // Return a Monarch syntax declaration, for use with the Monaco editor.
  // https://microsoft.github.io/monaco-editor/monarch.html

  static get monarchTokensProvider() {
    return {
      defaultToken: 'invalid',

      brackets: [
        ['{', '}', 'delimiter.curly'],
        ['[', ']', 'delimiter.square'],
        ['(', ')', 'delimiter.parenthesis'],
      ],

      // Common token patterns
      ws: /[ \t\r\n]*/,
      string: /"(?:[^\\\n\r"]|\\.)*"|'(?:[^\\\n\r']|\\.)*'/,
      number: /NaN\b|Infinity\b|-Infinity\b|-?\d+(\.\d+)?([eE]-?\d+)?/,
      boolean: /true\b|false\b/,
      suffix: /u8\b|u32\b|u64\b|i8\b|i32\b|i64\b|f16\b|f32\b/,
      identifier: /[A-Za-z]\w*/,

      tokenizer: {
        root: [
          {include: '@whitespace'},
          {include: '@comment'},

          // Assignment
          ['(@identifier)(@ws)(=)', ['variable.name', 'white', 'operator']],

          {include: '@expr'},
        ],

        // Expression
        expr: [
          {include: '@whitespace'},
          {include: '@comment'},

          // Number
          ['@number', 'number.float', '@suffix'],

          // Array
          [/\[/, '@brackets', '@array'],

          // String
          ['@string', 'string'],

          // Boolean
          ['@boolean', 'keyword'],

          // Dictionary
          [/{/, '@brackets', '@dict'],

          // Function invocation
          [
            '(@identifier)(@ws)(\\()',
            [
              'identifier',
              'white',
              {token: '@brackets', next: '@func'},
            ],
          ],

          // Identifier
          ['@identifier', 'identifier'],

          // Delimited subexpression
          [/\(/, '@brackets', '@subexpr'],

          // operators
          [/==|<=|<|>=|>|\+|-|\*|\/|\^|!/, 'operator'],
        ],

        // Function call
        func: [
          {include: '@expr'},
          [/,/, 'delimiter'],
          [/\)/, '@brackets', '@pop'],
        ],

        // Dictionary
        dict: [
          {include: '@whitespace'},
          {include: '@comment'},
          ['@string', 'string', '@propdef'],
          ['@identifier', 'identifier', '@propdef'],
          [/,/, 'delimiter'],
          [/}/, '@brackets', '@pop'],
        ],

        propdef: [
          {include: '@whitespace'},
          {include: '@comment'},
          [':', {token: 'delimiter', switchTo: '@propvalue'}],
        ],

        propvalue: [
          {include: '@expr'},
          [/,/, 'delimiter', '@pop'],
          [/(?=})/, '', '@pop'],
        ],

        // Array
        array: [
          {include: '@expr'},
          [/,/, 'delimiter'],
          [']', {token: '@brackets', switchTo: '@suffix'}],
        ],

        // Delimited subexpression
        subexpr: [
          {include: '@expr'},
          [/\)/, '@brackets', '@pop'],
        ],

        whitespace: [
          [/[ \t\r\n]+/, 'white'],
        ],

        comment: [
          [/(#|\/\/).*$/, 'comment'],
        ],

        suffix: [
          [
            '(@ws)((?:@suffix)?)',
            ['white', {token: 'annotation', next: '@pop'}],
          ],
        ],
      },
    };
  }
}
