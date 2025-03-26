const CACHE_NAME = 'soccer-prediction-v1';
const STATIC_CACHE_NAME = 'soccer-prediction-static-v1';
const DYNAMIC_CACHE_NAME = 'soccer-prediction-dynamic-v1';

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/favicon.ico',
  '/logo192.png',
  '/logo512.png'
];

// Install a service worker
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(STATIC_CACHE_NAME)
      .then(cache => {
        console.log('Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  const cacheWhitelist = [STATIC_CACHE_NAME, DYNAMIC_CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('Service Worker activated');
      return self.clients.claim();
    })
  );
});

// Advanced cache strategy: stale-while-revalidate for most requests
self.addEventListener('fetch', event => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }

  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // For HTML navigation requests - network first with offline fallback
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Clone the response before using it and storing it in the cache
          const responseClone = response.clone();
          caches.open(DYNAMIC_CACHE_NAME)
            .then(cache => {
              cache.put(event.request, responseClone);
            });
          return response;
        })
        .catch(() => {
          return caches.match(event.request)
            .then(response => {
              return response || caches.match('/offline.html');
            });
        })
    );
    return;
  }

  // For API requests - network only
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // If offline, return a basic JSON response for API calls
          return new Response(
            JSON.stringify({ 
              error: 'You are currently offline. Please try again when you have an internet connection.' 
            }),
            { 
              headers: { 'Content-Type': 'application/json' } 
            }
          );
        })
    );
    return;
  }

  // For asset requests - cache first, then network
  if (
    event.request.url.match(/\.(js|css|png|jpg|jpeg|gif|svg|ico)$/) ||
    event.request.url.includes('/static/')
  ) {
    event.respondWith(
      caches.match(event.request)
        .then(cachedResponse => {
          if (cachedResponse) {
            // Even if we have a cached version, fetch an update for next time
            fetch(event.request)
              .then(networkResponse => {
                caches.open(DYNAMIC_CACHE_NAME)
                  .then(cache => {
                    cache.put(event.request, networkResponse.clone());
                  });
              })
              .catch(() => console.log('Failed to update cache for:', event.request.url));
            
            return cachedResponse;
          }

          // If not in cache, fetch from network and cache
          return fetch(event.request)
            .then(networkResponse => {
              const responseToCache = networkResponse.clone();
              caches.open(DYNAMIC_CACHE_NAME)
                .then(cache => {
                  cache.put(event.request, responseToCache);
                });
              return networkResponse;
            })
            .catch(() => {
              // If both cache and network fail, return a generic fallback
              if (event.request.url.match(/\.(png|jpg|jpeg|gif|svg)$/)) {
                return caches.match('/logo192.png');
              }
              // For other resources, just fail
              return new Response('Resource not available offline', {
                status: 404,
                headers: { 'Content-Type': 'text/plain' }
              });
            });
        })
    );
    return;
  }

  // Default strategy - stale-while-revalidate
  event.respondWith(
    caches.open(DYNAMIC_CACHE_NAME).then(cache => {
      return cache.match(event.request).then(cachedResponse => {
        const fetchPromise = fetch(event.request)
          .then(networkResponse => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          })
          .catch(() => console.log('Failed to fetch:', event.request.url));
        
        return cachedResponse || fetchPromise;
      });
    })
  );
});

// Background sync for offline actions
self.addEventListener('sync', event => {
  if (event.tag === 'sync-predictions') {
    event.waitUntil(syncPredictions());
  }
});

// Function to sync predictions when back online
async function syncPredictions() {
  try {
    const dbName = 'offlinePredictions';
    const openRequest = indexedDB.open(dbName, 1);
    
    const db = await new Promise((resolve, reject) => {
      openRequest.onsuccess = () => resolve(openRequest.result);
      openRequest.onerror = () => reject(openRequest.error);
    });
    
    const transaction = db.transaction('predictions', 'readonly');
    const store = transaction.objectStore('predictions');
    
    const predictions = await new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    
    if (predictions.length === 0) return;
    
    // Send each prediction to the server
    for (const prediction of predictions) {
      try {
        const response = await fetch('/api/predictions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(prediction)
        });
        
        if (response.ok) {
          // Remove from IndexedDB if successfully sent to server
          const deleteTransaction = db.transaction('predictions', 'readwrite');
          const deleteStore = deleteTransaction.objectStore('predictions');
          deleteStore.delete(prediction.id);
        }
      } catch (error) {
        console.error('Failed to sync prediction:', error);
      }
    }
  } catch (error) {
    console.error('Error syncing predictions:', error);
  }
}

// Handle push notifications
self.addEventListener('push', event => {
  if (!event.data) {
    console.log('Push event but no data');
    return;
  }

  try {
    const data = event.data.json();
    const options = {
      body: data.body || 'New update from Soccer Prediction System',
      icon: '/logo192.png',
      badge: '/logo192.png',
      vibrate: [100, 50, 100],
      data: {
        url: data.url || '/'
      }
    };

    event.waitUntil(
      self.registration.showNotification(data.title || 'Soccer Prediction Update', options)
    );
  } catch (error) {
    console.error('Error showing notification:', error);
  }
});

// Handle notification click
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  const url = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({
      type: 'window',
      includeUncontrolled: true
    }).then(windowClients => {
      // Check if there is already a window/tab open with the target URL
      for (let i = 0; i < windowClients.length; i++) {
        const client = windowClients[i];
        if (client.url === url && 'focus' in client) {
          return client.focus();
        }
      }
      
      // If no open window/tab with the URL, open a new one
      if (clients.openWindow) {
        return clients.openWindow(url);
      }
    })
  );
}); 