// This file derives from https://js.tensorflow.org/js/codeSnippets.js

// There isn't a standard way to get line numbers reliably, we attempt to
// parse out a few formats we have seen.
// Note we do not use e.lineNumber as it is not reliable (especially in the
// context of eval)
function getLineNumber(error) {
  try {
    // firefox
    const firefoxRegex = /eval:(\d+):\d+/;
    if (error.stack.match(firefoxRegex)) {
      const res = error.stack.match(firefoxRegex);
      return parseInt(res[1], 10);
    }

    // chrome
    const chromeRegex = /eval.+:(\d+):\d+/;
    if (error.stack.match(chromeRegex)) {
      const res = error.stack.match(chromeRegex);
      return parseInt(res[1], 10);
    }

  } catch (e) {
    return;
  }

  // We found nothing
  return;
}
    
export async function executeCodeSnippet(consoleLogElement, codeSnippet) {
  consoleLogElement.innerText = '';
  var oldLog = console.log;
  console.log = function(...values) {
    let logStrs = [];
    for (let i = 0; i < values.length; i++) {
      const value = values[i];

      let logStr;
      if (value.toString == null) {
        logStr = value;
      } else {
        const toStr = value.toString();

        if (toStr === '[object Object]') {
          logStr = JSON.stringify(value, null, 2);
        } else {
          logStr = toStr;
        }
        logStrs.push(logStr);
      }
    }
    consoleLogElement.innerHTML += logStrs.join(' ') + '<br>';
  };

  function reportError(e) {
    var errorMessage = '\n<div class="snippet-error"><em>An error occured';
    var lineNumber = getLineNumber(e);
    if (lineNumber !== undefined) {
      errorMessage += ' on line: ' + lineNumber + '</em>';
    } else {
      errorMessage += '</em>'
    }
    errorMessage += '<br/>';
    errorMessage += '<div class="snippet-error-msg">' + e.message + '</div>';
    errorMessage += '</div>';

    console.log(errorMessage);
  }

  // It is important that codeSnippet and 'try {' be on the same line
  // in order to not modify the line number on an error.
  const evalString = '(async function runner() { try { ' + codeSnippet +
      '} catch (e) { reportError(e); } })()';

  // this outer try is for errors that prevent the snippet from being parsed.
  try {
    await eval(evalString).catch(function(e) {
      // This catch is for errors within promises within snippets
      reportError(e);
    });
  } catch (e) {
    reportError(e);
  }

  console.log = oldLog;
};
