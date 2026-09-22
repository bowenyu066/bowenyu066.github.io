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
  fetch(`${origin}/counter/${encodeURIComponent(path)}.json`, {
    credentials: 'omit',
    signal: AbortSignal.timeout(8000),
  }).then(response => {
    if (!response.ok) throw new Error('Counter unavailable');
    return response.json();
  }).then(data => {
    // A failure or a new, unindexed page must never look like a genuine zero.
    if (typeof data.count !== 'string' || !/^[0-9][0-9, .\u00a0\u202f]*$/.test(data.count)) return;
    const chinese = document.documentElement.lang.startsWith('zh');
    counter.textContent = chinese ? `${data.count} 次浏览` : `${data.count} views`;
    counter.title = chinese ? '浏览次数定期更新' : 'View count updates periodically';
    counter.hidden = false;
  }).catch(() => { /* Analytics must never interrupt the page. */ });
})();
