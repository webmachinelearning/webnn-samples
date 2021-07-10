import {samplesRepo} from './samples_repo.js';
import {sizeOfShape} from '../common/utils.js';

window.sizeOfShape = sizeOfShape;

export function main() {
  const selectElement = document.getElementById('example-select');
  for (const name of samplesRepo.names()) {
    const option = document.createElement('option');
    option.innerHTML = name;
    selectElement.appendChild(option);
  }
  const codeEditorElement = document.getElementById('code-editor');
  // eslint-disable-next-line new-cap
  const codeEditor = CodeMirror(codeEditorElement, {
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
    samplesRepo.getCode(selectElement.value).then((code) => {
      codeEditor.setOption('value', code);
      logElement.innerHTML = '';
    });
  });

  // Handle example earch parameter.
  const searchParams = new URLSearchParams(location.search);
  const exampleName = searchParams.get('example');
  if (Array.from(samplesRepo.names()).includes(exampleName)) {
    selectElement.value = exampleName;
  }
  selectElement.dispatchEvent(new Event('change'));
}
