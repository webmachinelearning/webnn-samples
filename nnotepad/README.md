# What is this?

**NNotepad** is a browser-based playground for experimenting with [WebNN](https://webmachinelearning.github.io/webnn/) expressions without boilerplate code. As of mid-2024, WebNN is available as a prototype in Chromium-based browsers, but requires launching the browser with particular flags enabled.


# Usage

Type assignments like `foo = 1 + 2` or expressions like `2 * foo`. The result of the last assignment or expression is shown. Some examples:

```
1 + 2
# yields 3

a = 123
b = 456
a / b
# yields 0.2697368562221527

A = [[1,7],[2,4]]
B = [[3,3],[5,2]]
matmul(A,B)
# yields [[38,17],[26,14]]
```

**NNotepad** translates what you type into script that builds a WebNN graph, evaluates the script, then executes the graph. Click 🔎 to see the generated script.

Expressions can use:

* Operators `+`, `-`, `*`, `/`, `^`, `==`, `<`, `<=`, `>`, `>=`, `!` with precedence, and `(`,`)` for grouping.
* Function calls like `add()`, `matmul()`, `sigmoid()`, and so on.
* Numbers like `-12.34`.
* Tensors like `[[1,2],[3,4]]`.
* Dictionaries like `{alpha: 2, beta: 3}`, arrays like `[ A, B ]`, strings like `"float32"`, and booleans `true` and `false`.

Functions and operators are turned into [`MLGraphBuilder`](https://webmachinelearning.github.io/webnn/#mlgraphbuilder) method calls.

Array literals (`[...]`) and number literals (`12.34`) are interpreted contextually:

* In assignments, they are intepreted as tensor/scalar constants [`MLOperand`](https://webmachinelearning.github.io/webnn/#mloperand)s, e.g. `alpha = 12.34` (scalar) or `T = [1,2,3,4]` (tensor).
* As arguments in function calls, they are interpreted depending on the argument definition, e.g. `neg(123)` (scalar), `neg([1,2,3])` (tensor), `concat([A,B,C],0)` (number).
* In options dictionaries inside function calls, they are interpreted depending on the dictionary definition. e.g. `linear(123, {alpha: 456, beta: 789})` (numbers), `transpose(T, {permutation: [0,2,1]})` (array of numbers), `gemm(A, B, {c: 123})` (scalar), `gemm(A, B, {c: [123]})` (tensor).
* In dictionaries outside of function calls, they are interpreted as arrays/numbers, e.g. `options = {alpha: 456, beta: 789})`. To pass a tensor/scalar constant in a dictionary, use a variable or wrap it in [`identity()`](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-identity) e.g. `options = {c:identity(4)}  gemm(A, B, options)`.

The default [data type](https://webmachinelearning.github.io/webnn/#enumdef-mloperanddatatype) for scalars and tensors is [`float32`](https://webmachinelearning.github.io/webnn/#dom-mloperanddatatype-float32). To specify a different data type, suffix with one of `i8`, `u8`, `i32`, `u32`, `i64`, `u64`, `f16`, `f32`, e.g. `123i8` or `[1,2,3]u32`.


# Helpers

In addition to WebNN [`MLGraphBuilder`](https://webmachinelearning.github.io/webnn/#mlgraphbuilder) methods, you can use these helpers:

* **load(_url_, _shape_, _dataType_)** - fetch a tensor resource. Must be served with appropriate [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) headers. Example: `load('https://www.random.org/cgi-bin/randbyte?nbytes=256', [16, 16], 'uint8')`
* **zeros(_shape_, _dataType_)** - constant zero-filled tensor of the given shape. Example: `zeros([2,2,2,2], 'int8')`


# Details & Gotchas

* [`float16`](https://webmachinelearning.github.io/webnn/#dom-mloperanddatatype-float16) support (and the `f16` suffix) is experimental.
* Whitespace including line breaks is ignored.
* Parsing around the "unary minus" operator can be surprising. Wrap expressions e.g. `(-a)` if you get unexpected errors.
* If output is a constant, it will be wrapped with [`identity()`](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-identity) if your back-end supports it. Otherwise, you must introduce a supported expression.

What ops are supported, and with what data types, depends entirely on your browser's WebNN implementation. Here be dragons!


# Parsing & Grammar

```
Anything after # or // on a line is ignored (outside other tokens)

{} means 0-or-more repetitions
[] means 0-or-1 repetitions
() for grouping
| separates options
'' is literal
// is regex

program = line { line }
line = assigment | expr
assigment = identifier '=' expr

expr = relexpr
relexpr = addexpr { ( '==' | '<' | '<=' | '>' | '>=' ) addexpr }
addexpr = mulexpr { ( '+' | '-' ) mulexpr }
mulexpr = powexpr { ( '*' | '/' ) powexpr }
powexpr = unyexpr { '^' unyexpr }
unyexpr = ( '-' | '!' ) unyexpr
        | finexpr
finexpr = number [ suffix ]
        | array [ suffix ]
        | string
        | boolean
        | dict
        | identifier [ '(' [ expr { ',' expr } ] ')' ]
        | '(' expr ')'

string = /("([^\\\x0A\x0D"]|\\.)*"|'([^\\\x0A\x0D']|\\.)*')/
number = /NaN|Infinity|-Infinity|-?\d+(\.\d+)?([eE]-?\d+)?/
boolean = 'true' | 'false'
identifier = /[A-Za-z]\w*/
suffix = 'u8' | 'u32' | 'i8' | 'i32' | 'u64' | 'i64' | 'f16' | 'f32'

array = '[' [ expr { ',' expr } ] ']'

dict = '{' [ propdef { ',' propdef  } [ ',' ] ] '}'
propdef = ( identifier | string ) ':' expr
```

