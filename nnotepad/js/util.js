class Util {
  static debounce(func, delay) {
    let timeoutId = 0;
    return function() {
      const $this = this, $arguments = arguments;
      if (timeoutId)
        clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        func.apply($this, $arguments);
        timeoutId = 0;
      }, delay);
    };
  }

  // Like JSON.stringify(), but handles BigInts, NaN, +/-Infinity, and -0
  static stringify(value) {
    let json = JSON.stringify(value, (k, v) => {
      if (typeof v === 'bigint')
        return 'ℝ:' + String(v) + 'n';
      if (Object.is(v, NaN))
        return 'ℝ:NaN';
      if (Object.is(v, Infinity))
        return 'ℝ:Infinity';
      if (Object.is(v, -Infinity))
        return 'ℝ:-Infinity';
      if (Object.is(v, -0))
        return 'ℝ:-0';
      return v;
    });
    return json.replaceAll(/"ℝ:(.*?)"/g, '$1');
  }

  static stringifyNumber(value, dataType) {
    if (dataType === 'int64' || dataType === 'uint64') {
      return String(value) + 'n';
    }
    return String(value);
  }

  static async loadBuffer(url) {
    const request = await fetch(url);
    if (!request.ok)
      throw new Error(`load failed ${request.statusText}`);
    return await request.arrayBuffer();
  }
}
