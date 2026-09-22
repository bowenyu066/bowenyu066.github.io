(() => {
  const config = window.siteAnalytics;
  if (!config?.endpoint || location.hostname !== config.hostname) return;
  // Speaker previews embed the presentation; neither preview should inflate counts.
  if (window.self !== window.top || ['presenter', 'preview'].some(key => new URLSearchParams(location.search).has(key))) return;
  let endpoint;
  try {
    endpoint = new URL(config.endpoint);
    if (endpoint.protocol !== 'https:' || !/^[a-z0-9-]+\.goatcounter\.com$/.test(endpoint.hostname)) return;
  } catch { return; }
  const origin = endpoint.origin;
  const path = location.pathname.replace(/\/index\.html$/, '/');
  window.goatcounter = {
    // A single key for / and /index.html; query strings and slide hashes are omitted.
    path,
    allow_frame: false,
  };
  const tracker = document.createElement('script');
  tracker.async = true;
  tracker.src = 'https://gc.zgo.at/count.js';
  tracker.dataset.goatcounter = `${origin}/count`;
  document.head.append(tracker);

  const counter = document.querySelector('[data-page-views]');
  if (!counter) return;
  const chinese = document.documentElement.lang.startsWith('zh');
  counter.hidden = false;
  counter.textContent = chinese ? '浏览次数加载中…' : 'Loading views…';
  // Explicit all-time range also avoids the initially cached, empty default response.
  fetch(`${origin}/counter/${encodeURIComponent(path)}.json?start=2000-01-01`, {
    credentials: 'omit',
    signal: AbortSignal.timeout(8000),
  }).then(response => {
    if (!response.ok) throw new Error(response.status === 404 ? 'pending' : 'unavailable');
    return response.json();
  }).then(data => {
    // Invalid responses must never look like a genuine zero.
    if (typeof data.count !== 'string' || !/^[0-9][0-9, .\u00a0\u202f]*$/.test(data.count)) throw new Error('unavailable');
    counter.textContent = chinese ? `${data.count} 次浏览` : `${data.count} views`;
    counter.title = chinese ? '浏览次数定期更新' : 'View count updates periodically';
    counter.hidden = false;
  }).catch(error => {
    counter.textContent = error.message === 'pending'
      ? (chinese ? '浏览次数待更新' : 'Views pending')
      : (chinese ? '浏览次数暂不可用' : 'Views unavailable');
  });
})();
