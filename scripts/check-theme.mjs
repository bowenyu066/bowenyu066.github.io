import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Exercise the scripts shipped in the built page with a controllable local clock.
const html = readFileSync('dist/index.html', 'utf8');
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)]
  .map(match => match[1])
  .filter(source => /themeOverrideKey|giscusScript|data-theme-toggle/.test(source));
assert.equal(scripts.length, 3);

function page(time, { saved = {}, blocked = false } = {}) {
  let now = new Date(time).getTime();
  class Clock extends Date {
    constructor(...args) { super(...(args.length ? args : [now])); }
    static now() { return now; }
  }
  const store = new Map(Object.entries(saved));
  const listeners = { window: {}, document: {}, button: {} };
  const listen = target => (name, handler) => { listeners[target][name] = handler; };
  const attributes = {};
  const button = { setAttribute: (key, value) => { attributes[key] = value; }, addEventListener: listen('button') };
  const lightIcon = {};
  const darkIcon = {};
  const root = { dataset: {} };
  const comments = [];
  const messages = [];
  let iframe;
  let timer;
  const document = {
    documentElement: root,
    body: { dataset: { page: '' } },
    currentScript: { parentElement: { append: script => comments.push(script) } },
    createElement: () => ({ dataset: {} }),
    addEventListener: listen('document'),
    hidden: false,
    querySelector: selector => ({
      '[data-theme-toggle]': button,
      '[data-theme-icon="light"]': lightIcon,
      '[data-theme-icon="dark"]': darkIcon,
      'iframe.giscus-frame': iframe,
    })[selector],
  };
  const context = vm.createContext({
    Date: Clock, document,
    window: { addEventListener: listen('window'), scrollY: 0 },
    localStorage: {
      getItem: key => { if (blocked) throw Error('Storage blocked'); return store.get(key) ?? null; },
      setItem: (key, value) => { if (blocked) throw Error('Storage blocked'); store.set(key, value); },
    },
    setTimeout: (callback, delay) => { timer = { callback, delay }; return 1; },
    clearTimeout: () => { timer = undefined; },
  });
  scripts.forEach(source => vm.runInContext(source, context));
  return {
    root, comments, messages, attributes, store, document, lightIcon, darkIcon,
    setTime: time => { now = new Date(time).getTime(); },
    tick: () => timer.callback(),
    delay: () => timer.delay,
    click: () => listeners.button.click(),
    focus: () => listeners.window.focus(),
    visible: () => listeners.document.visibilitychange(),
    storage: event => listeners.window.storage(event),
    ready: () => {
      iframe = { contentWindow: { postMessage: (message, origin) => messages.push({ message, origin }) } };
      listeners.window.message({ origin: 'https://giscus.app', source: iframe.contentWindow });
    },
    message: event => listeners.window.message(event ?? { origin: 'https://giscus.app', source: iframe.contentWindow }),
  };
}

for (const [time, expected] of [
  ['00:00:00', 'dark'], ['06:59:59', 'dark'], ['07:00:00', 'light'],
  ['12:00:00', 'light'], ['18:59:59', 'light'], ['19:00:00', 'dark'], ['23:59:59', 'dark'],
]) {
  const view = page(`2026-10-14T${time}`);
  assert.equal(view.root.dataset.theme, expected, time);
  assert.equal(view.comments[0].dataset.theme, expected, `Giscus at ${time}`);
  assert.equal(view.lightIcon.hidden, expected === 'dark');
  assert.equal(view.darkIcon.hidden, expected === 'light');
}

for (const options of [{ saved: { theme: 'dark' } }, { blocked: true }, { saved: { 'theme-override': '{bad json' } }]) {
  assert.equal(page('2026-10-14T12:00:00', options).root.dataset.theme, 'light');
}

const view = page('2026-10-14T18:59:59');
assert.equal(view.delay(), 1000);
view.ready();
assert.equal(view.messages.at(-1).message.giscus.setConfig.theme, 'light');
view.setTime('2026-10-14T19:00:00');
view.tick();
assert.equal(view.root.dataset.theme, 'dark');
assert.equal(view.messages.at(-1).message.giscus.setConfig.theme, 'dark');
assert.equal(view.messages.at(-1).origin, 'https://giscus.app');
const messageCount = view.messages.length;
view.message();
view.message({ origin: 'https://example.com', source: {} });
view.tick();
assert.equal(view.messages.length, messageCount, 'No repeated Giscus configuration messages');

view.click();
assert.equal(view.root.dataset.theme, 'light');
assert.equal(view.messages.at(-1).message.giscus.setConfig.theme, 'light');
assert.equal(view.attributes['aria-label'], 'Switch to dark mode');
const saved = Object.fromEntries(view.store);
assert.equal(page('2026-10-14T23:00:00', { saved }).root.dataset.theme, 'light', 'Override survives navigation');
assert.equal(page('2026-10-15T07:00:00', { saved }).root.dataset.theme, 'light');
view.setTime('2026-10-15T08:00:00');
view.focus();
view.setTime('2026-10-15T19:00:00');
view.visible();
assert.equal(view.root.dataset.theme, 'dark', 'Expired override resumes the clock');

const daytime = page('2026-10-14T12:00:00');
daytime.click();
assert.equal(daytime.root.dataset.theme, 'dark');
daytime.setTime('2026-10-15T07:00:00');
daytime.tick();
assert.equal(daytime.root.dataset.theme, 'light');
daytime.storage({ key: 'theme-override', newValue: JSON.stringify({ theme: 'dark', expiresAt: new Date('2026-10-15T19:00:00').getTime() }) });
assert.equal(daytime.root.dataset.theme, 'dark', 'Manual choice syncs across tabs');
daytime.storage({ key: null, newValue: null });
assert.equal(daytime.root.dataset.theme, 'light');

const unavailable = page('2026-10-14T12:00:00', { blocked: true });
unavailable.click();
assert.equal(unavailable.root.dataset.theme, 'dark');
unavailable.setTime('2026-10-15T07:00:00');
unavailable.tick();
assert.equal(unavailable.root.dataset.theme, 'light');

// Overnight boundaries remain local 7 a.m. across daylight-saving transitions.
for (const date of ['2026-03-07', '2026-10-31']) {
  const overnight = page(`${date}T23:00:00`);
  overnight.click();
  const override = JSON.parse(overnight.store.get('theme-override'));
  assert.equal(new Date(override.expiresAt).getHours(), 7);
}
console.log(`Theme checks passed (${Intl.DateTimeFormat().resolvedOptions().timeZone}); clock, overrides, Giscus, and storage covered.`);
