/* service-worker.js — app-shell caching for the Gestura PWA.

   IMPORTANT: the live webcam stream (/video_feed, MJPEG) and the state stream
   (/state, SSE) are INFINITE responses — they must never pass through the cache
   or the SW would hang. Those, plus the POST-y endpoints, are left to the
   network untouched. Everything static (HTML shell, CSS, JS, icons, fonts) is
   cached so the app installs and its shell loads offline.

   Note: "works offline" means the UI shell loads. Live translation still needs
   the Python server (that's where the camera + model run). */

const CACHE = 'gestura-v6';
const SHELL = [
  '/',
  '/static/style.css',
  '/static/sequence.css',
  '/static/sequence.js',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/manifest.webmanifest',
];

// endpoints that must ALWAYS hit the network (streams / dynamic)
const NETWORK_ONLY = ['/video_feed', '/state', '/polish', '/clear', '/practice', '/service-worker.js'];

function cacheable(url) {
  return url.origin === location.origin ||
         url.host.endsWith('googleapis.com') ||
         url.host.endsWith('gstatic.com');
}

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;                 // POST/etc → straight to network

  const url = new URL(req.url);

  // live streams / dynamic endpoints: no SW involvement at all
  if (url.origin === location.origin &&
      NETWORK_ONLY.some((p) => url.pathname.startsWith(p))) {
    return;
  }

  // navigations: network-first so you get fresh HTML, fall back to cached shell
  if (req.mode === 'navigate') {
    e.respondWith((async () => {
      try {
        const res = await fetch(req);
        const c = await caches.open(CACHE);
        c.put('/', res.clone());
        return res;
      } catch (_) {
        return (await caches.match('/')) || Response.error();
      }
    })());
    return;
  }

  // static assets (css/js/icons/fonts): network-first so edits always show when
  // online; fall back to cache when offline. (Was cache-first, which pinned
  // stale JS/CSS after every change and required manual cache-busting.)
  e.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res && res.status === 200 && cacheable(url)) {
        const c = await caches.open(CACHE);
        c.put(req, res.clone());
      }
      return res;
    } catch (_) {
      return (await caches.match(req)) || Response.error();
    }
  })());
});
