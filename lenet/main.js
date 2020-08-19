const predictButton = document.getElementById('predict');
const nextButton = document.getElementById('next');
const clearButton = document.getElementById('clear');
const visualCanvas = document.getElementById('visual_canvas');
const visualContext = visualCanvas.getContext('2d');
const digitCanvas = document.createElement('canvas');
digitCanvas.setAttribute('height', 28);
digitCanvas.setAttribute('width', 28);
digitCanvas.style.backgroundColor = 'black';
const digitContext = digitCanvas.getContext('2d');

function drawNextDigitFromMnist() {
  const n = Math.floor(Math.random() * 10);
  const digit = mnist[n].get();
  mnist.draw(digit, digitContext);
  visualContext.drawImage(digitCanvas, 0, 0, visualCanvas.width, visualCanvas.height);
}

function getInputFromCanvas() {
  digitContext.clearRect(0, 0, digitCanvas.width, digitCanvas.height);
  digitContext.drawImage(visualCanvas, 0, 0, digitCanvas.width, digitCanvas.height);
  const imageData = digitContext.getImageData(0, 0, digitCanvas.width, digitCanvas.height);
  const input = new Float32Array(digitCanvas.width * digitCanvas.height);
  for (var i = 0; i < input.length; i++) {
    input[i] = imageData.data[i * 4];
  }
  return input;
}

function clearResult() {
  for (let i = 0; i < 3; ++i) {
    let labelElement = document.getElementById(`label${i}`);
    let probElement = document.getElementById(`prob${i}`);
    labelElement.innerHTML = '';
    probElement.innerHTML = '';
  }
}

async function main() {
  drawNextDigitFromMnist();
  let pen = new Pen(visualCanvas);
  const lenet = new Lenet('lenet.bin');
  try {
    let start = performance.now();
    await lenet.load();
    console.log(`loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    start = performance.now();
    await lenet.compile();
    console.log(`compilation elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    predictButton.removeAttribute('disabled');
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
  predictButton.addEventListener('click', async function (e) {
    try {
      const input = getInputFromCanvas();
      let start = performance.now();
      const result = await lenet.predict(input);
      console.log(`execution elapsed time: ${(performance.now() - start).toFixed(2)} ms`);
      console.log(`execution result: ${result}`);
      const classes = topK(result);
      classes.forEach((c, i) => {
        console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
        let labelElement = document.getElementById(`label${i}`);
        let probElement = document.getElementById(`prob${i}`);
        labelElement.innerHTML = `${c.label}`;
        probElement.innerHTML = `${c.prob}%`;
      });
    } catch (error) {
      console.log(error);
      addWarning(error.message);
    }
  });
  nextButton.addEventListener('click', () => {
    drawNextDigitFromMnist();
    clearResult();
  });

  clearButton.addEventListener('click', () => {
    pen.clear();
    clearResult();
  })
}

function topK(probs, k = 3) {
  const sorted = probs.map((prob, index) => [prob, index]).sort((a, b) => {
    if (a[0] === b[0]) {
      return 0;
    }
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();

  const classes = [];
  for (let i = 0; i < k; ++i) {
    let c = {
      label: sorted[i][1],
      prob: (sorted[i][0] * 100).toFixed(2)
    }
    classes.push(c);
  }

  return classes;
}

function addWarning(msg) {
  let div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  let container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

function removeWarning() {
  $('.alert').alert('close')
}

