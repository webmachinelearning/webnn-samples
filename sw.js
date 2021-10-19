// Use service worker to enable SharedArrayBuffer on GitHub page.
// Refer to https://dev.to/stefnotch/enabling-coop-coep-without-touching-the-server-2d3n
// Disabled this in Electron.js environment as which sets Headers
// in another way.
if (typeof module !== 'object') {
  if (typeof window === 'undefined') {
    self.addEventListener('install', () => self.skipWaiting());

    self.addEventListener('activate', (event) => {
      event.waitUntil(self.clients.claim());
    });

    self.addEventListener('fetch', (event) => {
      if (event.request.cache === 'only-if-cached' &&
          event.request.mode !== 'same-origin') {
        return;
      }

      event.respondWith(fetch(event.request).then((response) => {
        if (response.status === 0) return response;
        const newHeaders = new Headers(response.headers);
        newHeaders.set('Cross-Origin-Embedder-Policy', 'require-corp');
        newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');

        const moddedResponse = new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });

        return moddedResponse;
      }).catch((e) => console.error(e)));
    });
  } else {
    (() => {
      if ('serviceWorker' in navigator) {
        const src = window.document.currentScript.src;
        // Register service worker
        navigator.serviceWorker.register(src).then((registration) => {
          console.log('COOP/COEP Service Worker registered',
              registration.scope);

          registration.addEventListener('updatefound', () => {
            window.location.reload();
          });

          // If the registration is active, but it's not controlling the page
          if (registration.active && !navigator.serviceWorker.controller) {
            window.location.reload();
          }
        },
        (err) => {
          console.log('COOP/COEP Service Worker failed to register', err);
        });
      } else {
        console.warn('Cannot register a service worker');
      }
    })();
  }
}
