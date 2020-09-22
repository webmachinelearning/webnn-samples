'use stricit';

export class Pen {
  constructor(cavans) {
    this.canvas = cavans;
    this.canvas.style.backgroundColor = 'black';
    this.canvas.style.cursor = 'crosshair';
    this.context = cavans.getContext('2d');
    this.down = false;
    this.start = {};
    const self = this;
    this.canvas.addEventListener('mousedown', (e) => {
      self.down = true;
      self.start = self.getPosition(e);
    });
    this.canvas.addEventListener('mouseup', (e) => {
      self.down = false;
    });
    this.canvas.addEventListener('mousemove', (e) => {
      if (self.down) {
        const end = self.getPosition(e);
        self.draw(self.start, end);
        self.start = end;
      }
    });
  }

  getPosition(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return {x: x, y: y};
  }

  draw(start, end) {
    this.context.strokeStyle = 'white';
    this.context.lineJoin = 'round';
    this.context.lineWidth = 20;

    this.context.beginPath();
    this.context.moveTo(start.x, start.y);
    this.context.lineTo(end.x, end.y);
    this.context.closePath();
    this.context.stroke();
  }

  clear() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
