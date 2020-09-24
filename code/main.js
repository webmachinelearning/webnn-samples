import {executeCodeSnippet} from './code_snippet.js';
import {samplesMap} from './samples_map.js';

window.sizeOfShape = function(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
};

export function main() {
  const selectElement = document.getElementById('example-select');
  for (const sample of samplesMap) {
    const option = document.createElement('option');
    option.innerHTML = sample[0];
    selectElement.appendChild(option);
  }
  const codeEditorElement = document.getElementById('code-editor');
  // eslint-disable-next-line new-cap
  const codeEditor = CodeMirror(codeEditorElement, {
    value: samplesMap.get(selectElement.value),
    theme: 'railscasts',
    mode: 'javascript',
    tabSize: 2,
    indentUnit: 2,
    viewportMargin: Infinity,
    lineNumbers: true,
    lineWrapping: true,
    keyMap: 'sublime',
    readOnly: true,
  });

  const runButton = document.getElementById('run');
  const logElement = document.getElementById('console-log');
  runButton.addEventListener('click', function() {
    executeCodeSnippet(logElement, codeEditor.getValue());
  });

  const editButton = document.getElementById('edit');
  editButton.addEventListener('click', function() {
    codeEditor.setOption('readOnly', false);
  });

  selectElement.addEventListener('change', function() {
    codeEditor.setOption('value', samplesMap.get(selectElement.value));
    logElement.innerHTML = '';
  });
}
