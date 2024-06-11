import {Util} from './util.js';
import {NNotepad, ParseError} from './nnotepad.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
document.addEventListener('DOMContentLoaded', async (e) => {
  try {
    const req = await fetch('res/default.txt');
    if (req.ok) {
      $('#input').value = await req.text();
    }
  } catch (ex) {
    console.warn(ex);
  }

  async function refresh(e) {
    const code = $('#input').value;
    $('#output').innerText = '';
    $('#output').style.color = '';
    $('#srcText').innerText = '';

    if (!code.trim()) {
      return;
    }

    try {
      const [builderFunc, src] = NNotepad.makeBuilderFunction(code);
      $('#srcText').innerText = src;
      const result =
          await NNotepad.execBuilderFunction($('#device').value, builderFunc);
      $('#output').innerText = explain(result);
    } catch (ex) {
      $('#output').style.color = 'red';
      $('#output').innerText = ex.name + ': ' + ex.message;
      if (!(ex instanceof ParseError || ex instanceof ReferenceError)) {
        let tip = '(See console for more)';
        if (ex.message.match(/read properties of undefined/)) {
          tip = 'Maybe WebNN is not supported by your browser?';
        } else if (ex.message.match(/is not an output operand/)) {
          tip = 'Tip: Try wrapping expression with identity()';
        }

        $('#output').innerText += '\n\n' + tip;
        console.warn(ex);
      }
    }
  }

  $('#input').addEventListener('input', Util.debounce(refresh, 500));
  $('#device').addEventListener('change', refresh);

  refresh();

  $$('dialog > button').forEach((e) => e.addEventListener('click', (e) => {
    e.target.parentElement.close();
  }));
  $$('dialog').forEach((dialog) => {
    dialog.addEventListener('click', (e) => {
      const rect = e.target.getBoundingClientRect();
      if (e.clientY < rect.top || e.clientY > rect.bottom ||
          e.clientX < rect.left || e.clientX > rect.right) {
        e.target.close();
      }
    });
    dialog.addEventListener('close', (e) => {
      $('#input').focus();
    });
  });
  $('#peek').addEventListener('click', (e) => $('#srcDialog').showModal());
  $('#help').addEventListener('click', (e) => $('#helpDialog').showModal());

  $('#resize').addEventListener('pointerdown', (e) => {
    const resize = e.target;
    resize.setPointerCapture(e.pointerId);
    const listener = (e) => {
      document.documentElement.style.setProperty(
          '--input-height', `${e.clientY}px`);
    };
    resize.addEventListener('pointermove', listener);
    resize.addEventListener('pointerup', () => {
      resize.releasePointerCapture(e.pointerId);
      resize.removeEventListener('pointermove', listener);
    }, {once: true});
  });
});

function explain(outputs) {
  return outputs
      .map(
          (output) => ['dataType: ' + output.dataType,
            'shape: ' + Util.stringify(output.shape),
            'tensor: ' + dumpTensor(output.shape, output.buffer, 8),
          ].join('\n'))
      .join('\n\n');


  function dumpTensor(shape, buffer, indent) {
    // Scalar
    if (shape.length === 0) {
      return String(buffer[0]);
    }

    const width = [...buffer]
        .map((n) => String(n).length)
        .reduce((a, b) => Math.max(a, b), 0);

    const out = [];
    let bufferIndex = 0;

    return (function convert(dim = 0) {
      out.push('[');
      for (let i = 0; i < shape[dim]; ++i) {
        if (dim + 1 === shape.length) {
          out.push(String(buffer[bufferIndex++]).padStart(width));
        } else {
          convert(dim + 1);
        }
        if (i !== shape[dim] - 1) {
          out.push(', ');
          if (dim + 1 !== shape.length) {
            out.push('\n'.repeat(shape.length - dim - 1));
            out.push(' '.repeat(indent + dim + 1));
          }
        }
      }
      out.push(']');
      return out.join('');
    })();
  }
}

