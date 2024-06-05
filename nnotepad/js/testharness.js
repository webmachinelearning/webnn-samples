// ============================================================
// Test Harness
// ============================================================

// Harness.section(description) - start a section (required)
// Harness.ok(message) - add a success to current section
// Harness.error(message) - add a failure to current section

class Harness { // eslint-disable-line no-unused-vars
  static section(s) {
    Harness.current = {
      details: document.createElement('details'),
      summary: Object.assign(document.createElement('summary'), {innerText: s}),
      counts:
          Object.assign(document.createElement('div'), {className: 'counts'}),
      pass: 0,
      fail: 0,
    };
    Harness.current.summary.append(Harness.current.counts);
    Harness.current.details.append(Harness.current.summary);
    document.body.append(Harness.current.details);
  }

  static updateCounts() {
    Harness.current.counts.innerText =
        `pass: ${Harness.current.pass} / fail: ${Harness.current.fail}`;
  }

  static log(s, options) {
    Harness.current.details.append(
        Object.assign(document.createElement('div'), options, {innerText: s}));
  }

  static ok(s) {
    Harness.log(s, {});
    Harness.current.pass += 1;
    Harness.updateCounts();
  }

  static error(s) {
    Harness.log(s, {className: 'failure'});
    Harness.current.fail += 1;
    Harness.updateCounts();
    Harness.current.details.open = true;
  }
}
