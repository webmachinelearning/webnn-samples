### How to build the `process` WASM files?
Download and install [Emscripten](https://emscripten.org/) follow the [guide](https://emscripten.org/docs/getting_started/downloads.html). Emscripten compiles C and C++ to WebAssembly using LLVM and Binaryen. Emscripten output can run on the Web, in Node.js, and in wasm runtimes. 

Download the [code](https://github.com/miaobin/rnnoise.git) which is based on [RNNoise](https://github.com/xiph/rnnoise) and enter the `examples` folder. Extract the pre-processed and post-processed parts of the project and write the code following the `rnnoise_demo.c`.

Download the project and checkout branch:
```
git clone https://github.com/miaobin/rnnoise.git && cd rnnoise
git checkout -b wasm_process origin/wasm_process
```

Generate the WASM file:
```
cd examples
emcc -g -O3 -s ALLOW_MEMORY_GROWTH=1 -s EXPORT_ALL=1 -s EXPORTED_RUNTIME_METHODS="['cwrap']" -s EXPORTED_FUNCTIONS="['_malloc', '_free']" -I ../include/ ../src/*.c ./process.c -o process/process.js
```
When using emcc to build to WebAssembly, you will see a `.wasm `file containing that code, as well as the usual `.js` file that is the main target of compilation. Those two are built to work together: run the `.js` file, and it will load and set up the WebAssembly code for you, properly setting up imports and exports for it, etc.

### How to use the `process` WASM library?
Load the `process.js` file. And the `Module.onRuntimeInitialized` function will be called when the runtime is fully initialized. Use `Module.cwrap` function to return `pre_processing` and `post_processing` JavaScript wrapper for native C functions. The `processer.js` file is an example of the above steps. You can easily call the pre-processing and post-processing functions.

### License
This RNNoise project is licensed under the BSD 3-Clause "New" or "Revised" License. [Learn more](https://choosealicense.com/licenses/bsd-3-clause/).