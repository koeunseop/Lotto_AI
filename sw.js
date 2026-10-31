/* 오프라인 실행용 서비스 워커: 앱 파일과 파이썬 엔진(Pyodide)을 처음 받은 뒤에는 기기에 저장해 두고 쓴다 */
const SHELL = 'lotto-shell-v1';
const CDN = 'lotto-cdn-v1';
const FILES = ['./', 'index.html', 'app.js', 'worker.js', 'engine_mobile.py', 'seed_draws.json',
               'manifest.webmanifest', 'icon-192.png', 'icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== SHELL && k !== CDN).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    // 앱 파일: 저장본을 바로 보여 주고, 뒤에서 최신본으로 갱신한다
    e.respondWith(
      caches.open(SHELL).then(async (cache) => {
        const hit = await cache.match(req, { ignoreSearch: true });
        const fresh = fetch(req).then((res) => { if (res.ok) cache.put(req, res.clone()); return res; }).catch(() => hit);
        return hit || fresh;
      })
    );
  } else if (url.hostname === 'cdn.jsdelivr.net') {
    // 파이썬 엔진·라이브러리: 한 번 받으면 계속 저장본을 쓴다
    e.respondWith(
      caches.open(CDN).then(async (cache) => {
        const hit = await cache.match(req);
        if (hit) return hit;
        const res = await fetch(req);
        if (res.ok) cache.put(req, res.clone());
        return res;
      })
    );
  }
});
