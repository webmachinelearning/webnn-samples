export function drawKeyPoints(image, canvas, keyPoints, rects) {
  const ctx = canvas.getContext('2d');
  rects.forEach((rect, n) => {
    const keyPoint = keyPoints[n];
    for (let i = 0; i < 136; i = i + 2) {
      // decode keyPoint
      const x =
          (rect[2] * keyPoint[i] + rect[0]) / image.height * canvas.height;
      const y =
          (rect[3] * keyPoint[i + 1] + rect[1]) / image.height * canvas.height;
      // draw keyPoint
      ctx.beginPath();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.closePath();
    }
  });
}
