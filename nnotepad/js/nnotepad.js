/* global BigInt64Array, BigUint64Array, Float16Array */
/* global Util */

// ============================================================
// General Utilities
// ============================================================

class ParseError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ParseError';
  }
}

class BuildError extends Error {
  constructor(message) {
    super(message);
    this.name = 'build()';
  }
}

class ComputeError extends Error {
  constructor(message) {
    super(message);
    this.name = 'compute()';
  }
}

// ============================================================
// General WebNN Utilities
// ============================================================

class WebNNUtil {
  static bufferForOperand(operand) {
    const size = [...operand.shape()].reduce((a, b) => a * b, 1);
    const ctor = WebNNUtil.dataTypeToBufferType(operand.dataType());
    return new ctor(size); // eslint-disable-line new-cap
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

  static isNonOperandArg(name, index) {
    return ({
      concat: [0, 1],
      expand: [1],
      gru: [3, 4],
      gruCell: [4],
      lstm: [3, 4],
      lstmCell: [5],
      pad: [1, 2],
      reshape: [1],
      slice: [1, 2],
      softmax: [1], // TODO: Distinguish overloads
      split: [1],
    })[name]
        ?.includes(index);
  }
}

class NNotepad { // eslint-disable-line no-unused-vars
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
      '!':
          'logicalNot', // See
      // https://github.com/webmachinelearning/webnn/issues/496#issuecomment-2123895106
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
        'NaN|Infinity|-Infinity|-?\\d+(\\.\\d+)?([eE]-?\\d+)?';
    const kStringPattern =
        `"([^\\\\\\x0A\\x0D"]|\\\\.)*"|'([^\\\\\\x0A\\x0D']|\\\\.)*'`;
    const kBooleanPattern = 'true|false';
    const kSuffixPattern = `u8|u32|u64|i8|i32|i64|f16|f32`;
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
    return [new AsyncFunction(['_'], src), src];

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
    function serializeExpr(expr, nonOperand = false) {
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
          return nonOperand ? Util.stringify(expr.value) :
                              serializeScalar(expr.value, expr.dataType);
        case 'array':
          return nonOperand ? serializeArray(expr.value) :
                              serializeTensor(expr.value, expr.dataType);
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
                return `${k}: ${serializeExpr(v, true)}`;
              })
              .join(', ') +
          '}';
    }

    function serializeScalar(number, dataType) {
      const ctor = WebNNUtil.dataTypeToBufferType(dataType);
      return `_.constant({dataType:"${dataType}"}, new ${ctor.name}([${
        Util.stringifyNumber(number, dataType)}]))`;
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
      const dimensions = [];
      const elements = [];
      (function measure(t, d) {
        if (d >= dimensions.length) {
          dimensions[d] = t.length;
        } else if (dimensions[d] !== t.length) {
          throw new Error('Invalid tensor: inconsistent dimensions');
        }
        t.forEach((e) => {
          if (e.type === 'array') {
            measure(e.value, d + 1);
          } else if (e.type !== 'number') {
            throw new Error(`Invalid tensor: saw ${e.type}`);
          } else if (d + 1 !== dimensions.length) {
            throw new Error('Invalid tensor: saw scalar');
          } else {
            elements.push(e.value);
          }
        });
      }(tensor, 0));
      const ctor = WebNNUtil.dataTypeToBufferType(dataType);
      return `_.constant({dataType: "${dataType}", dimensions: ${
        Util.stringify(dimensions)}}, new ${ctor.name}([${
        elements.map((n) => Util.stringifyNumber(n, dataType)).join(',')}]))`;
    }

    function serializeArray(array) {
      return '[' + array.map((expr) => serializeExpr(expr)).join(', ') + ']';
    }

    function serializeCall(name, args) {
      if (name === 'load') {
        const [url, shape, dataType] = args;
        if (url.type !== 'string') {
          throw new TypeError('load(): expected string');
        }
        if (shape.type !== 'tensor') {
          throw new TypeError('load(): expected array');
        }
        if (dataType.type !== 'string') {
          throw new TypeError('load(): expected string');
        }
        const ctor = WebNNUtil.dataTypeToBufferType(dataType.value);
        return `_.constant({dataType: "${dataType.value}", dimensions: ${
          Util.stringify(shape.value)}}, new ${
          ctor.name}(await Util.loadBuffer(${Util.stringify(url.value)})))`;
      }

      return `_.${name}(${
        args.map(
            (arg, index) => serializeExpr(
                arg, WebNNUtil.isNonOperandArg(name, index)))
            .join(', ')})`;
    }
  }

  // ============================================================
  // Script Executor
  // ============================================================

  // Call with the output of `makeBuilderFunc()`. Builds an MLContext and
  // MLGraphBuilder, executes the function to make an MLGraph, then runs
  // compute() on it. The output is mapped.

  static async execBuilderFunction(deviceType, builderFunc) {
    const context = await navigator.ml.createContext({deviceType});
    const builder = new self.MLGraphBuilder(context);

    const outputOperands = [];
    let output = await builderFunc(builder);
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
    const outputBuffers = {};
    outputOperands.forEach((op, index) => {
      const name = `output-${index}`;
      namedOutputs[name] = op;
      outputBuffers[name] = WebNNUtil.bufferForOperand(op);
    });

    let graph;
    try {
      graph = await builder.build(namedOutputs);
    } catch (ex) {
      console.warn(ex);
      throw new BuildError(`${ex.name} : ${ex.message}`);
    }
    const inputBuffers = {};

    let result;
    try {
      result = await context.compute(graph, inputBuffers, outputBuffers);
    } catch (ex) {
      console.warn(ex);
      throw new ComputeError(`${ex.name} : ${ex.message}`);
    }

    function maybeProxyForFloat16Array(array) {
      return ('proxyForFloat16Array' in self) ?
          self.proxyForFloat16Array(array) :
          array;
    }

    // window.result = result;
    // console.log(result);
    return outputOperands.map(
        (op, index) => ({
          dataType: op.dataType(),
          shape: op.shape(),
          buffer: maybeProxyForFloat16Array(result.outputs[`output-${index}`]),
        }));
  }
}
