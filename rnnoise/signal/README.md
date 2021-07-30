### How to build the `signal` WASM files?
Download and install [Emscripten](https://emscripten.org/) follow the [guide](https://emscripten.org/docs/getting_started/downloads.html). Emscripten compiles C and C++ to WebAssembly using LLVM and Binaryen. Emscripten output can run on the Web, in Node.js, and in wasm runtimes. 

Download [RNNoise](https://github.com/xiph/rnnoise) and enter the `examples` folder. Extract the pre-processed and post-processed parts of the project and write the code following the `rnnoise_demo.c`.

Generate the WASM file:
```
emcc -g -O3 -s ALLOW_MEMORY_GROWTH=1 -s EXPORT_ALL=1 -s EXPORTED_RUNTIME_METHODS='["cwrap"]' -I ../include/ ../src/*.c ../examples/your_code.c -o signal.js
```
When using emcc to build to WebAssembly, you will see a `.wasm `file containing that code, as well as the usual `.js` file that is the main target of compilation. Those two are built to work together: run the `.js` file, and it will load and set up the WebAssembly code for you, properly setting up imports and exports for it, etc.

### How to use the `signal` WASM library?
Load the `signal.js` file. And the `Module.onRuntimeInitialized` function will be called when the runtime is fully initialized. Use `Module.cwrap` function to return `pre_processing` and `post_processing` JavaScript wrapper for native C functions. The `processer.js` file is an example of the above steps. You can easily call the pre-processing and post-processing functions.

### License
This RNNoise project is licensed under the BSD 3-Clause "New" or "Revised" License. [Learn more](https://choosealicense.com/licenses/bsd-3-clause/).