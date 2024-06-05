// Hacky "polyfill" for Float16Array
// See "Approach" notes below for behavior and limitations.

(function(global) {
'use strict';

if ('Float16Array' in global)
  return;

// Based on https://github.com/inexorabletash/polyfill/blob/master/typedarray.js
function packIEEE754(v, ebits, fbits) {
  var bias = (1 << (ebits - 1)) - 1;

  function roundToEven(n) {
    var w = Math.floor(n), f = n - w;
    if (f < 0.5)
      return w;
    if (f > 0.5)
      return w + 1;
    return w % 2 ? w + 1 : w;
  }

  // Compute sign, exponent, fraction
  var s, e, f;
  if (v !== v) {
    // NaN
    // http://dev.w3.org/2006/webapi/WebIDL/#es-type-mapping
    e = (1 << ebits) - 1;
    f = Math.pow(2, fbits - 1);
    s = 0;
  } else if (v === Infinity || v === -Infinity) {
    e = (1 << ebits) - 1;
    f = 0;
    s = (v < 0) ? 1 : 0;
  } else if (v === 0) {
    e = 0;
    f = 0;
    s = (1 / v === -Infinity) ? 1 : 0;
  } else {
    s = v < 0;
    v = Math.abs(v);

    if (v >= Math.pow(2, 1 - bias)) {
      // Normalized
      e = Math.min(Math.floor(Math.log(v) / Math.LN2), 1023);
      var significand = v / Math.pow(2, e);
      if (significand < 1) {
        e -= 1;
        significand *= 2;
      }
      if (significand >= 2) {
        e += 1;
        significand /= 2;
      }
      var d = Math.pow(2, fbits);
      f = roundToEven(significand * d) - d;
      e += bias;
      if (f / d >= 1) {
        e += 1;
        f = 0;
      }
      if (e > 2 * bias) {
        // Overflow
        e = (1 << ebits) - 1;
        f = 0;
      }
    } else {
      // Denormalized
      e = 0;
      f = roundToEven(v / Math.pow(2, 1 - bias - fbits));
    }
  }

  // Pack sign, exponent, fraction
  var bits = [], i;
  for (i = fbits; i; i -= 1) {
    bits.push(f % 2 ? 1 : 0);
    f = Math.floor(f / 2);
  }
  for (i = ebits; i; i -= 1) {
    bits.push(e % 2 ? 1 : 0);
    e = Math.floor(e / 2);
  }
  bits.push(s ? 1 : 0);
  bits.reverse();
  var str = bits.join('');

  // Bits to bytes
  var bytes = [];
  while (str.length) {
    bytes.unshift(parseInt(str.substring(0, 8), 2));
    str = str.substring(8);
  }
  return bytes;
}

function unpackIEEE754(bytes, ebits, fbits) {
  // Bytes to bits
  var bits = [], i, j, b, str, bias, s, e, f;

  for (i = 0; i < bytes.length; ++i) {
    b = bytes[i];
    for (j = 8; j; j -= 1) {
      bits.push(b % 2 ? 1 : 0);
      b = b >> 1;
    }
  }
  bits.reverse();
  str = bits.join('');

  // Unpack sign, exponent, fraction
  bias = (1 << (ebits - 1)) - 1;
  s = parseInt(str.substring(0, 1), 2) ? -1 : 1;
  e = parseInt(str.substring(1, 1 + ebits), 2);
  f = parseInt(str.substring(1 + ebits), 2);

  // Produce number
  if (e === (1 << ebits) - 1) {
    return f !== 0 ? NaN : s * Infinity;
  } else if (e > 0) {
    // Normalized
    return s * Math.pow(2, e - bias) * (1 + f / Math.pow(2, fbits));
  } else if (f !== 0) {
    // Denormalized
    return s * Math.pow(2, -(bias - 1)) * (f / Math.pow(2, fbits));
  } else {
    return s < 0 ? -0 : 0;
  }
}

function unpackF16(b) {
  return unpackIEEE754(b, 5, 10);
}
function packF16(v) {
  return packIEEE754(v, 5, 10);
}
function f16ToU16(u16) {
  const [lo, hi] = packF16(u16);
  return lo | (hi << 8);
}
function u16ToF16(u16) {
  return unpackF16([u16 & 0xFF, (u16 >> 8) & 0xFF]);
}

function isArrayIndex(s) {
  return s === String(Number(s) | 0);
}

function makeProxy(target) {
  return new Proxy(target, {
    get(target, property) {
      if (property === Symbol.iterator) {
        return function*() {
          for (let u16 of target) {
            yield u16ToF16(u16);
          }
        };
      } else if (
        typeof property === 'string' && isArrayIndex(property)) {
        const u16 = target[property];
        return typeof u16 === 'number' ? u16ToF16(u16) : u16;
      } else {
        return target[property];
      }
    },
    set(target, property, value, receiver) {
      if (typeof property === 'string' && isArrayIndex(property)) {
        target[property] = f16ToU16(value);
        return true;
      } else {
        return Reflect.set(target, property, value, receiver);
      }
    }
  });
}

// Approach #1: subclass Uint16Array, with a Proxy
// * Pro: `instanceof Float16Array` works
// * Con: Not recognized as an ArrayBufferView by DOM methods
global.Float16Array = class Float16Array extends Uint16Array {
  constructor(...args) {
    if (Array.isArray(args[0])) {
      const array = args[0];
      super(array.length);
      for (let i = 0; i < array.length; ++i) {
        this[i] = f16ToU16(array[i]);
      }
    } else {
      super(...args);
    }

    return makeProxy(this);
  }
};


// Approach #2: Proxy for Uint16Array
// * Pro: Can extract target
// * Con: Not recognized as an ArrayBufferView by DOM methods
global.Float16Array = function Float16Array(...args) {
  let target;
  if (Array.isArray(args[0])) {
    const array = args[0];
    target = new Uint16Array(array.length);
    for (let i = 0; i < array.length; ++i) {
      this[i] = f16ToU16(array[i]);
    }
  } else {
    target = new Uint16Array(...args);
  }

  return makeProxy(target);
};


// Approach #3: Return Uint16Array with getters/setters
// * Pro: Can pass to DOM methods
// * Con: Fails, as the indexed properties are not configurable!
global.Float16Array = function Float16Array(...args) {
  let target;
  if (Array.isArray(args[0])) {
    const array = args[0];
    target = new Uint16Array(array.length);
    for (let i = 0; i < array.length; ++i) {
      this[i] = f16ToU16(array[i]);
    }
  } else {
    target = new Uint16Array(...args);
  }

  const proxy = new Uint16Array(target.buffer);
  for (let property = 0; property < target.length; ++property) {
    proxy.__defineGetter__(property, () => {
      return u16ToF16(target[property]);
    });

    proxy.__defineSetter__(property, value => {
      target[property] = f16ToU16(value);
    });
  }
  return proxy;
};


// Approach #4: Separate ctor and proxy helpers
//
// Construction is done with `new Float16Array(...)` but a plain Uint16Array
// is returned, initialized with the passed float16 data.
//
// To read values, call `proxy = proxyForFloat16(array)` and then use
// the proxy instead of the original for all use of the array. If the
// passed array is not a Uint16Array it just returns it. Note that if
// Float16Array is not a polyfill then this will **not** be added to the
// global, so check that the function exists before using it!
global.Float16Array = function Float16Array(arg, ...rest) {
  let target;
  if (arg instanceof ArrayBuffer) {
    throw new Error('Constructing from ArrayBuffer not supported');
  } else if (typeof arg === 'object') {
    const arrayLike = arg;
    const length = Number(arrayLike.length);
    target = new Uint16Array(length);
    const proxy = makeProxy(target);
    for (let index = 0; index < length; ++index) {
      proxy[index] = arrayLike[index];
    }
    return target;
  } else {
    const length = Number(arg);
    return new Uint16Array(length);
  }
};
global.proxyForFloat16Array = function(target) {
  if (!(target instanceof Uint16Array))
    return target;

  return makeProxy(target);
};

})(self);
