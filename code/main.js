import {executeCodeSnippet} from './code_snippet.js';
import {sampleCode} from './sample_code.js';

export function main() {
  const codeEditorElement = document.getElementById('code-editor');
  // eslint-disable-next-line new-cap
  const codeEditor = CodeMirror(codeEditorElement, {
    value: sampleCode,
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
  runButton.addEventListener('click', function() {
    const logElement = document.getElementById('console-log');
    executeCodeSnippet(logElement, codeEditor.getValue());
  });

  const editButton = document.getElementById('edit');
  editButton.addEventListener('click', function() {
    codeEditor.setOption('readOnly', false);
  });
}
