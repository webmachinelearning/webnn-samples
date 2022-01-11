// This file is used for postprocessing predicted result from
// Tiny Yolo V2 models, which is referenced from
// https://github.com/experiencor/keras-yolo2/blob/master/utils.py and licensed
// under https://github.com/experiencor/keras-yolo2/blob/master/LICENSE
class BoundBox {
  constructor(xmin, ymin, xmax, ymax, confidence = 0, classes = []) {
    this.xmin = xmin;
    this.ymin = ymin;
    this.xmax = xmax;
    this.ymax = ymax;

    this.confidence = confidence;
    this.classes = classes;

    this.label = -1;
    this.score = -1;
  }

  getLabel() {
    if (this.label === -1) {
      let max = 0;
      let index = 0;
      for (let i = 0; i < this.classes.length; ++i) {
        if (this.classes[i] > max) {
          max = this.classes[i];
          index = i;
        }
      }
      this.label = index;
    }
    return this.label;
  }

  getScore() {
    if (this.score === -1) {
      this.score = this.classes[this.getLabel()];
    }
    return this.score;
  }
}

export function decodeYOLOv2(options, output, anchors) {
  const {
    numClasses = 20,
    numBoxes = 5,
    gridHeight = 13,
    gridWidth = 13,
    objThreshold = 0.5,
    nmsThreshold = 0.3,
  } = options;
  // (x, y, w, h) + confidence + classes
  const size = 4 + 1 + numClasses;

  // Decode the output by the network
  for (let i = 0; i < gridHeight * gridWidth * numBoxes; ++i) {
    output[size * i + 4] = sigmoid(output[size * i + 4]);
  }

  const classes = [];
  const indexes = [];

  for (let i = 0; i < gridHeight * gridWidth * numBoxes; ++i) {
    let classesI =
        output.slice((numClasses + 5) * i + 5, (numClasses + 5) * (i + 1));
    classesI = softmax(classesI);
    let isOutputClass = false;
    for (let j = 0; j < numClasses; ++j) {
      const tmp = (output[size * i + 4] * classesI[j]);
      classesI[j] = 0;
      if (tmp > objThreshold) {
        classesI[j] = tmp;
        isOutputClass = true;
      }
    }
    classes.push(classesI);
    if (isOutputClass) indexes.push(i);
  }

  // Get bounding boxes
  const boxes = [];
  indexes.forEach((index) => {
    const classI = classes[index];
    const b = index % numBoxes;
    const col = (index - b) / numBoxes % gridWidth;
    const row = ((index - b) / numBoxes - col) / gridWidth % gridHeight;
    let x = output[size * index + 0];
    let y = output[size * index + 1];
    let w = output[size * index + 2];
    let h = output[size * index + 3];
    // center position, unit: image width
    x = (col + sigmoid(x)) / gridWidth;
    // center position, unit: image height
    y = (row + sigmoid(y)) / gridHeight;
    // unit: image width
    w = anchors[2 * b + 0] * Math.exp(w) / gridWidth;
    // unit: image height
    h = anchors[2 * b + 1] * Math.exp(h) / gridHeight;
    const confidence = output[size * index + 4];
    const box = new BoundBox(
        x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classI);
    boxes.push(box);
  });

  // Suppress non-maximal boxes (NMS)
  const tmpBoxes = [];
  let sortedBoxes = [];
  for (let c = 0; c < numClasses; ++c) {
    for (let i = 0; i < boxes.length; ++i) {
      tmpBoxes[i] = [boxes[i], i];
    }
    sortedBoxes = tmpBoxes.sort((a, b) => {
      return (b[0].classes[c] - a[0].classes[c]);
    });
    for (let i = 0; i < sortedBoxes.length; ++i) {
      if (sortedBoxes[i][0].classes[c] === 0) continue;
      else {
        for (let j = i + 1; j < sortedBoxes.length; ++j) {
          if (bboxIou(sortedBoxes[i][0], sortedBoxes[j][0]) >= nmsThreshold) {
            boxes[sortedBoxes[j][1]].classes[c] = 0;
          }
        }
      }
    }
  }

  // Remove the boxes which are less likely than a objThreshold
  const trueBoxes = [];
  boxes.forEach((box) => {
    if (box.getScore() > objThreshold) {
      trueBoxes.push(box);
    }
  });

  const result = [];

  for (let i = 0; i < trueBoxes.length; ++i) {
    if (Math.max(...trueBoxes[i].classes) === 0) continue;
    const predictedClassId = trueBoxes[i].getLabel();
    const score = trueBoxes[i].score;
    const a = (trueBoxes[i].xmax + trueBoxes[i].xmin) / 2;
    const b = (trueBoxes[i].ymax + trueBoxes[i].ymin) / 2;
    const c = (trueBoxes[i].xmax - trueBoxes[i].xmin);
    const d = (trueBoxes[i].ymax - trueBoxes[i].ymin);
    result.push([predictedClassId, a, b, c, d, score]);
  }

  return result;
}

export function getBoxes(results, margin) {
  const objBoxes = [];

  for (let i = 0; i < results.length; ++i) {
    // Display detected object
    const classId = results[i][0];
    const x = results[i][1];
    const y = results[i][2];
    const w = results[i][3];
    const h = results[i][4];
    const prob = results[i][5];
    const [xmin, xmax, ymin, ymax] = cropYoloBox(x, y, w, h, margin);
    objBoxes.push([classId, xmin, xmax, ymin, ymax, prob]);
  }

  return objBoxes;
}

export function drawBoxes(image, canvas, objBoxes, labels) {
  const ctx = canvas.getContext('2d');
  const imWidth = image.naturalWidth || image.width;
  const imHeight = image.naturalHeight || image.height;
  // drawImage
  canvas.width = imWidth / imHeight * canvas.height;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  // drawBox
  const colors =
      ['#ff0000', '#ffc107', '#00b067', '#704e99', '#ff3860', '#009bea'];
  objBoxes.forEach((box) => {
    const label = labels[box[0]];
    const xmin = box[1] * canvas.width;
    const xmax = box[2] * canvas.width;
    const ymin = box[3] * canvas.height;
    const ymax = box[4] * canvas.height;
    const prob = box[5];
    ctx.strokeStyle = colors[box[0] % colors.length];
    ctx.fillStyle = colors[box[0] % colors.length];
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
    ctx.font = '20px Arial';
    const text = `${label}: ${prob.toFixed(2)}`;
    const width = ctx.measureText(text).width;
    if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
      ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10),
          width + 4, parseInt(ctx.font, 10));
      ctx.fillStyle = 'white';
      ctx.textAlign = 'start';
      ctx.fillText(text, xmin, ymin - 3);
    } else {
      ctx.fillRect(xmin + 2, ymin, width + 4, parseInt(ctx.font, 10));
      ctx.fillStyle = 'white';
      ctx.textAlign = 'start';
      ctx.fillText(text, xmin + 2, ymin + 15);
    }
  });
}

function bboxIou(box1, box2) {
  const intersectW =
      intervalOverlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax]);
  const intersectH =
      intervalOverlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax]);
  const intersect = intersectW * intersectH;
  const w1 = box1.xmax - box1.xmin;
  const h1 = box1.ymax - box1.ymin;
  const w2 = box2.xmax - box2.xmin;
  const h2 = box2.ymax - box2.ymin;
  const union = w1 * h1 + w2 * h2 - intersect;
  return intersect / union;
}

function intervalOverlap(intervalA, intervalB) {
  const [x1, x2] = intervalA;
  const [x3, x4] = intervalB;

  if (x3 < x1) {
    if (x4 < x1) {
      return 0;
    } else {
      return Math.min(x2, x4) - x1;
    }
  } else {
    if (x2 < x3) {
      return 0;
    } else {
      return Math.min(x2, x4) - x3;
    }
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function softmax(arr) {
  const max = Math.max(...arr);
  let sum = 0;

  for (let i = 0; i < arr.length; ++i) {
    sum = Math.exp(arr[i] - max) + sum;
  }

  for (let i = 0; i < arr.length; ++i) {
    arr[i] = Math.exp(arr[i] - max) / sum;
  }

  return arr;
}

// Crop box
function cropYoloBox(x, y, w, h, margin) {
  let xmin = x - w / 2 * margin[0];
  let xmax = x + w / 2 * margin[1];
  let ymin = y - h / 2 * margin[2];
  let ymax = y + h / 2 * margin[3];
  if (xmin < 0) xmin = 0;
  if (ymin < 0) ymin = 0;
  if (xmax > 1) xmax = 1;
  if (ymax > 1) ymax = 1;
  return [xmin, xmax, ymin, ymax];
}
