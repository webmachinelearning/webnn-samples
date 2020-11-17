// JS port of https://github.com/microsoft/DNS-Challenge/blob/master/NSNet2-baseline/featurelib.py
/* eslint-disable camelcase */

export function calcFeat(Spec, cfg) {
  // """compute spectral features"""
  return tf.tidy(() => {
    let inpFeat;
    if (cfg['feattype'] === 'MagSpec') {
      inpFeat = tf.abs(Spec);
    } else if (cfg['feattype'] == 'LogPow') {
      const pmin = tf.scalar(10**(-12));
      const powSpec = tf.pow(tf.abs(Spec), 2);
      inpFeat = tf.maximum(powSpec, pmin);
      const data = inpFeat.dataSync();
      for (let i = 0; i < data.length; ++i) {
        data[i] = Math.log10(data[i]);
      }
      inpFeat = tf.tensor(data, Spec.shape);
    } else {
      throw new Error('Feature not implemented.');
    }
    return inpFeat;
  });
}

function hanningWindow(N) {
  const window = new Float32Array(N);
  for (let i = 0; i < N - 1; ++i) {
    window[i] = 0.5*(1 - Math.cos(6.283185307179586*i/(N-1)));
  }
  return tf.sqrt(window);
}

export function calcSpec(y, params, channel) {
  // """compute complex spectrum from audio file"""
  return tf.tidy(() => {
    const fs = parseInt(params['fs']);

    // STFT parameters
    const N_win = parseInt(parseFloat(params['winlen'])*fs);
    let N_fft;
    if ('nfft' in params) {
      N_fft = parseInt(params['nfft']);
    } else {
      N_fft = parseInt(parseFloat(params['winlen'])*fs);
    }
    const N_hop = parseInt(N_win * parseFloat(params['hopfrac']));
    const Y = tf.signal.stft(y, N_win, N_hop, N_fft, hanningWindow);
    return Y;
  });
}

export function spec2sig(Spec, params) {
  // """convert spectrum to time signal"""

  // # STFT parameters
  return tf.tidy(() => {
    const fs = parseInt(params['fs']);
    const N_win = parseInt(parseFloat(params['winlen'])*fs);
    let N_fft;
    if ('nfft' in params) {
      N_fft = parseInt(params['nfft']);
    } else {
      N_fft = parseInt(parseFloat(params['winlen'])*fs);
    }
    const N_hop = parseInt(N_win * parseFloat(params['hopfrac']));
    const win = hanningWindow(N_win);
    const x = istft(Spec, N_fft, win, N_hop);
    return x;
  });
}

function istft(X, N_fft, win, N_hop) {
  // """
  // inverse short-time Fourier transform
  //  X         Spectra [frequency x frames x channels]
  //  N_fft     FFT size (samples)
  //  win       window,  len(win) <= N_fft
  //  N_hop     hop size (samples)
  // """
  // # get lengths
  return tf.tidy(() => {
    // const specsize = X.shape[0];
    const N_frames = X.shape[1];
    if (X.rank < 3) {
      X = tf.expandDims(X, 2);
    }
    const M = X.shape[2];
    const N_win = win.shape[0];

    // # init
    const Nx = N_hop*(N_frames-1) + N_win;
    const win_M = tf.outerProduct(win, tf.ones([M]));
    let x = tf.zeros([Nx, M]);
    for (let nn = 0; nn < N_frames; ++nn) {
      const X_real = tf.real(X);
      const X_imag = tf.imag(X);
      const X_frame_real = tf.squeeze(tf.slice(X_real, [0, nn], [-1, 1]));
      const X_frame_imag = tf.squeeze(tf.slice(X_imag, [0, nn], [-1, 1]));
      const X_frame = tf.complex(X_frame_real, X_frame_imag);
      let x_win = tf.spectral.irfft(X_frame);
      x_win = x_win.reshape([N_fft, M]);
      x_win = tf.mul(win_M, tf.slice(x_win, [0], [N_win]));
      const idx1 = parseInt(nn*N_hop);
      const idx2 = parseInt(idx1+N_win);
      if (nn === N_frames - 1) {
        x = x.slice([0], [idx1]).concat(x_win.add(x.slice([idx1], [N_win])));
      } else {
        x = x.slice([0], [idx1]).concat(x_win.add(x.slice([idx1], [N_win])))
            .concat(x.slice([idx2]));
      }
    }

    if (M === 1) {
      x = tf.squeeze(x);
    }

    return x;
  });
}
