import { readFileSync, readdirSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source = readFileSync('public/analytics.js', 'utf8');
async function run({ endpoint = 'https://example.goatcounter.com', hostname = 'bowenyu066.github.io', search = '', lang = 'en', fail = false, status = 500, count = '1,234', framed = false } = {}) {
  const counter = { hidden: true };
  const scripts = [];
  const requests = [];
  const window = { siteAnalytics: { endpoint, hostname: 'bowenyu066.github.io' } };
  window.self = window;
  window.top = framed ? {} : window;
  vm.runInNewContext(source, {
    window, location: { hostname, pathname: '/blogs/example/index.html', search }, URL, URLSearchParams, AbortSignal,
    document: { documentElement: { lang }, createElement: () => ({ dataset: {} }), head: { append: s => scripts.push(s) }, querySelector: () => counter },
    fetch: async url => { requests.push(url); return { ok: !fail, status, json: async () => ({ count }) }; },
  });
  await new Promise(resolve => setImmediate(resolve));
  return { counter, scripts, requests, window };
}
const ok = await run();
assert.equal(ok.counter.textContent, '1,234 views');
assert.equal(ok.counter.hidden, false);
assert.equal(ok.window.goatcounter.path, '/blogs/example/');
assert.equal(ok.scripts[0].dataset.goatcounter, 'https://example.goatcounter.com/count');
assert.equal(ok.requests[0], 'https://example.goatcounter.com/counter/%2Fblogs%2Fexample%2F.json?start=2000-01-01');
assert.equal((await run({ lang: 'zh-CN' })).counter.textContent, '1,234 次浏览');
for (const options of [{ endpoint: '' }, { hostname: 'localhost' }, { endpoint: 'https://wrong.example' }, { search: '?presenter' }, { search: '?preview' }, { framed: true }]) {
  const result = await run(options);
  assert.equal(result.scripts.length, 0);
  assert.equal(result.requests.length, 0);
}
for (const options of [{ fail: true }, { count: '<img>' }, { count: null }]) assert.equal((await run(options)).counter.textContent, 'Views unavailable');
function htmlFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? htmlFiles(`${dir}/${entry.name}`) : entry.name.endsWith('.html') ? [`${dir}/${entry.name}`] : []);
}
const pages = htmlFiles('dist').filter(path => !path.startsWith('dist/sparky/'));
for (const path of pages) {
  const html = readFileSync(path, 'utf8');
  assert.equal((html.match(/src="\/analytics.js"/g) || []).length, 1, path);
  assert.ok(html.includes('data-page-views'), path);
}
console.log(`Analytics checks passed; ${pages.length} built pages covered (redirects excluded).`);

assert.equal((await run({fail: true, status: 404})).counter.textContent, 'Views pending');
assert.equal((await run({fail: true, lang: 'zh-CN'})).counter.textContent, '浏览次数暂不可用');
