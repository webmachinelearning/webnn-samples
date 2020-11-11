export class AudioPlayer {
  constructor(sampleRate, button, label) {
    this.sampleRate = sampleRate;
    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: this.sampleRate});
    this.source = null;
    this.button = button;
    this.label = label;
    this.updateButton('Play');
  }

  updateButton(state) {
    this.button.state = state;
    this.button.innerHTML = state + ' ' + this.label;
  }

  play(buffer) {
    this.source = this.audioCtx.createBufferSource();
    this.source.connect(this.audioCtx.destination);
    const audioBuffer = this.audioCtx.createBuffer(1, buffer.length, this.sampleRate);
    audioBuffer.copyToChannel(buffer, 0, 0);
    this.source.buffer = audioBuffer;
    this.source.start();
    this.audioCtx.resume();
    const self = this;
    this.source.onended = () => {
      self.updateButton('Play');
    }
    self.updateButton('Pause');
  }

  pause() {
    if (this.button.state === 'Pause') {
      try {
        this.audioCtx.suspend();
        this.updateButton('Resume');
      } catch(e) {
        console.log(e);
      }
    } 
  }

  resume() {
    try {
      this.audioCtx.resume();
      this.updateButton('Pause');
    } catch(e) {
      console.log(e);
    } 
  }

  stop() {
    try {
      if (this.source) {
        this.source.stop();
        this.source.disconnect(this.audioCtx.destination);
        this.source.onended = null;
        this.source = null;
        this.updateButton('Play');
      }
    } catch(e) {
      console.log(e);
    }
  }
}