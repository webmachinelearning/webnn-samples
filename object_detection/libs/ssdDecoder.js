// This file is used for postprocessing predicted result from
// SSD MobileNet V1 model, which is referenced from
// https://github.com/tensorflow/models/blob/master/research/object_detection/
// and licensed under Apache 2.0.

// Decode out box coordinate
// Referenced from
// https://github.com/tensorflow/models/blob/master/research/object_detection/box_coders/faster_rcnn_box_coder.py
export function decodeOutputBoxTensor(options, outputBoxTensor, anchors) {
  const {
    boxSize = 4,
    numBoxes = 1083 + 600 + 150 + 54 + 24 + 6,
  } = options;

  if (outputBoxTensor.length % boxSize !== 0) {
    throw new Error(
        `The length of outputBoxTensor should be the multiple of ${boxSize}!`);
  }

  // scaleFactors: [yScale, xScale, heightScale, widthScale]
  const scaleFactors = [10.0, 10.0, 5.0, 5.0];
  let boxOffset = 0;
  let ty;
  let tx;
  let th;
  let tw;
  let w;
  let h;
  let yCenter;
  let xCenter;

  for (let y = 0; y < numBoxes; ++y) {
    const [yCenterA, xCenterA, ha, wa] = anchors[y];
    ty = outputBoxTensor[boxOffset] / scaleFactors[0];
    tx = outputBoxTensor[boxOffset + 1] / scaleFactors[1];
    th = outputBoxTensor[boxOffset + 2] / scaleFactors[2];
    tw = outputBoxTensor[boxOffset + 3] / scaleFactors[3];
    w = Math.exp(tw) * wa;
    h = Math.exp(th) * ha;
    yCenter = ty * ha + yCenterA;
    xCenter = tx * wa + xCenterA;
    // Decoded box coordinate: [ymin, xmin, ymax, xmax]
    outputBoxTensor[boxOffset] = yCenter - h / 2;
    outputBoxTensor[boxOffset + 1] = xCenter - w / 2;
    outputBoxTensor[boxOffset + 2] = yCenter + h / 2;
    outputBoxTensor[boxOffset + 3] = xCenter + w / 2;
    boxOffset += boxSize;
  }
}

/*
* Get IOU(intersection-over-union) of 2 boxes
*
* @param {number[4]} boxCord1 - An 4 element Array of box coordinate.
* @param {number[4]} boxCord2 - An 4 element Array of box coordinate.
* @returns {number} IOU
*/
function iOU(boxCord1, boxCord2) {
  if (boxCord1.length !== 4 || boxCord2.length !== 4) {
    throw new Error('IOU: each input length should be 4!');
  }

  const [ymin1, xmin1, ymax1, xmax1] = boxCord1;
  const [ymin2, xmin2, ymax2, xmax2] = boxCord2;
  const minYmax = Math.min(ymax1, ymax2);
  const maxYmin = Math.max(ymin1, ymin2);
  const height = Math.max(0, minYmax - maxYmin);
  const minXmax = Math.min(xmax1, xmax2);
  const maxXmin = Math.max(xmin1, xmin2);
  const width = Math.max(0, minXmax - maxXmin);
  const intersection = height * width;
  const area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
  const area2 = (ymax2 - ymin2) * (xmax2 - xmin2);
  const areaSum = area1 + area2 - intersection;

  if (areaSum === 0) {
    throw new Error('[IOU] areaSum can not be 0!');
  }

  const IOU = intersection / areaSum;
  return IOU;
}

// Generate anchors
// Referenced from
// https://github.com/tensorflow/models/blob/master/research/object_detection/anchor_generators/multiple_grid_anchor_generator.py
// https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config
export function generateAnchors(options) {
  const {
    minScale = 0.2,
    maxScale = 0.95,
    aspectRatios = [1.0, 2.0, 0.5, 3.0, 0.3333],
    baseAnchorSize = [1.0, 1.0],
    featureMapShapeList = [[19, 19], [10, 10], [5, 5], [3, 3], [2, 2], [1, 1]],
    interpolatedScaleAspectRatio = 1.0,
    reduceBoxesInLowestLayer = true,
  } = options;
  const numLayers = featureMapShapeList.length;
  const boxSpecsList = [];
  const scales = [];

  for (let i = 0; i < numLayers; ++i) {
    const scale = minScale + (maxScale - minScale) * i / (numLayers - 1);
    scales.push(scale);
  }

  scales.forEach((scale, i) => {
    const scaleNext = (i === scales.length - 1) ? 1.0 : scales[i + 1];
    let layerBoxSpecs = [];
    if (i === 0 && reduceBoxesInLowestLayer) {
      layerBoxSpecs = [[0.1, 1.0], [scale, 2.0], [scale, 0.5]];
    } else {
      aspectRatios.forEach((aspectRatio, j) => {
        layerBoxSpecs.push([scale, aspectRatio]);
      });
      if (interpolatedScaleAspectRatio > 0.0) {
        layerBoxSpecs.push([Math.sqrt(scale * scaleNext),
          interpolatedScaleAspectRatio]);
      }
    }
    boxSpecsList.push(layerBoxSpecs);
  });

  const anchors = [];

  for (let i = 0; i < numLayers; ++i) {
    const gridHeight = featureMapShapeList[i][0];
    const gridWidth = featureMapShapeList[i][1];
    const anchorStride = [1.0 / gridHeight, 1.0 / gridWidth];
    const anchorOffset = [anchorStride[0] / 2, anchorStride[1] / 2];
    for (let h = 0; h < gridHeight; ++h) {
      for (let w = 0; w < gridWidth; ++w) {
        boxSpecsList[i].forEach((layerBoxSpec, j) => {
          const [scale, aspectRatio] = layerBoxSpec;
          const ratioSqrt = Math.sqrt(aspectRatio);
          const yCenter = h * anchorStride[0] + anchorOffset[0];
          const xCenter = w * anchorStride[1] + anchorOffset[1];
          const height = scale / ratioSqrt * baseAnchorSize[0];
          const width = scale * ratioSqrt * baseAnchorSize[0];
          anchors.push([yCenter, xCenter, height, width]);
        });
      }
    }
  }

  return anchors;
}

// NMS(Non Max Suppression)
// Referenced from
// https://github.com/tensorflow/models/blob/master/research/object_detection/core/post_processing.py#L38
export function nonMaxSuppression(
    options, outputBoxTensor, outputClassScoresTensor) {
  // Using a little higher threshold and lower max detections can save inference
  // time with little performance loss.
  const {
    scoreThreshold = 0.1, // 1e-8
    iouThreshold = 0.5,
    maxDetectionsPerClass = 10, // 100
    maxTotalDetections = 100,
    numBoxes = 1083 + 600 + 150 + 54 + 24 + 6,
    numClasses = 91,
    boxSize = 4,
  } = options;

  let totalDetections = null;
  let boxesList = [];
  let scoresList = [];
  let classesList = [];

  // Skip background 0
  for (let x = 1; x < numClasses; ++x) {
    let boxes = [];
    let scores = [];
    for (let y = 0; y < numBoxes; ++y) {
      const scoreIndex = y * numClasses + x;
      if (outputClassScoresTensor[scoreIndex] > scoreThreshold) {
        const boxIndexStart = y * boxSize;
        boxes.push(
            outputBoxTensor.subarray(boxIndexStart, boxIndexStart + boxSize));
        scores.push(outputClassScoresTensor[scoreIndex]);
      }
    }
    const boxForClassi = [];
    const scoreForClassi = [];
    const classi = [];
    while (scores.length !== 0 &&
        scoreForClassi.length < maxDetectionsPerClass) {
      let max = 0;
      let maxIndex = 0;
      // Find max score
      scores.forEach((score, j) => {
        if (score > max) {
          max = score;
          maxIndex = j;
        }
      });
      // Push and delete max
      const maxBox = boxes[maxIndex];
      boxForClassi.push(boxes.splice(maxIndex, 1)[0]);
      scoreForClassi.push(scores.splice(maxIndex, 1)[0]);
      classi.push(x);
      const retainBoxes = [];
      const retainScores = [];
      boxes.forEach((box, j) => {
        if (iOU(box, maxBox) < iouThreshold) {
          // Remain low IOU and delete high IOU
          retainBoxes.push(boxes[j]);
          retainScores.push(scores[j]);
        }
      });
      boxes = retainBoxes;
      scores = retainScores;
    }
    boxesList = boxesList.concat(boxForClassi);
    scoresList = scoresList.concat(scoreForClassi);
    classesList = classesList.concat(classi);
  }

  if (scoresList.length > maxTotalDetections) {
    totalDetections = maxTotalDetections;
    // quick sort to get max total detections
    let low = 0;
    let high = scoresList.length - 1;
    while (low < high) {
      let i = low;
      let j = high;
      const tmpScore = scoresList[i];
      const tmpBox = boxesList[i];
      const tmpClassi = classesList[i];
      while (i < j) {
        while (i < j && scoresList[j] < tmpScore) {
          --j;
        }
        if (i < j) {
          scoresList[i] = scoresList[j];
          boxesList[i] = boxesList[j];
          classesList[i] = classesList[j];
          ++i;
        }
        while (i < j && scoresList[i] > tmpScore) {
          ++i;
        }
        if (i < j) {
          scoresList[j] = scoresList[i];
          boxesList[j] = boxesList[i];
          classesList[j] = classesList[i];
          --j;
        }
      }
      scoresList[i] = tmpScore;
      boxesList[i] = tmpBox;
      classesList[i] = tmpClassi;
      if (i === maxTotalDetections) {
        low = high;
      } else if (i < maxTotalDetections) {
        low = i + 1;
      } else {
        high = i - 1;
      }
    }
  } else {
    totalDetections = scoresList.length;
  }

  return [totalDetections, boxesList, scoresList, classesList];
}


// Crop box
export function cropSsdBox(imageSource, totalDetections, boxesList, margin) {
  const imWidth = imageSource.naturalWidth || imageSource.width;
  const imHeight = imageSource.naturalHeight || imageSource.height;

  for (let i = 0; i < totalDetections; ++i) {
    const [ymin, xmin, ymax, xmax] = boxesList[i];
    boxesList[i][0] =
        Math.max(0, (ymax + ymin) / 2 - (ymax - ymin) / 2 * margin[2]);
    boxesList[i][2] =
        Math.min(imHeight, (ymax + ymin) / 2 + (ymax - ymin) / 2 * margin[3]);
    boxesList[i][1] =
        Math.max(0, (xmax + xmin) / 2 - (xmax - xmin) / 2 * margin[0]);
    boxesList[i][3] =
        Math.min(imWidth, (xmax + xmin) / 2 + (xmax - xmin) / 2 * margin[1]);
  }

  return boxesList;
}

// Draw img and box
export function drawBoxes(
    outputElement, totalDetections, imageSource, boxesList,
    scoresList, classesList, labels) {
  const ctx = outputElement.getContext('2d');
  const imWidth = imageSource.naturalWidth || imageSource.width;
  const imHeight = imageSource.naturalHeight || imageSource.height;
  outputElement.width = imWidth / imHeight * outputElement.height;
  const colors =
      ['#ff3860', '#ff0000', '#00b067', '#704e99', '#17a2b8', '#ffc107'];
  ctx.drawImage(imageSource, 0, 0, outputElement.width, outputElement.height);

  for (let i = 0; i < totalDetections; ++i) {
    // Skip background and blank
    const label = labels[classesList[i]];
    if (label !== '???') {
      let [ymin, xmin, ymax, xmax] = boxesList[i];
      ymin = Math.max(0, ymin);
      xmin = Math.max(0, xmin);
      ymax = Math.min(1, ymax);
      xmax = Math.min(1, xmax);
      ymin *= outputElement.height;
      xmin *= outputElement.width;
      ymax *= outputElement.height;
      xmax *= outputElement.width;
      const prob = 1 / (1 + Math.exp(-scoresList[i]));
      ctx.strokeStyle = colors[classesList[i] % colors.length];
      ctx.fillStyle = colors[classesList[i] % colors.length];
      ctx.lineWidth = 3;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
      ctx.font = '20px Arial';
      const text = `${label}: ${prob.toFixed(2)}`;
      const width = ctx.measureText(text).width;
      if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
        ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4,
            parseInt(ctx.font, 10));
        ctx.fillStyle = 'white';
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin, ymin - 3);
      } else {
        ctx.fillRect(xmin + 2, ymin, width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = 'white';
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin + 2, ymin + 15);
      }
    }
  }
}
