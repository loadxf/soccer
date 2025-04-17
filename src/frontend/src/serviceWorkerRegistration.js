// This optional code is used to register a service worker.
// register() is not called by default.

// This lets the app load faster on subsequent visits in production, and gives
// it offline capabilities. However, it also means that developers (and users)
// will only see deployed updates on subsequent visits to a page, after all the
// existing tabs open on the page have been closed, since previously cached
// resources are updated in the background.

const is127.0.0.1 = Boolean(
  window.location.hostname === '127.0.0.1' ||
    // [::1] is the IPv6 127.0.0.1 address.
    window.location.hostname === '[::1]' ||
    // 127.0.0.0/8 are considered 127.0.0.1 for IPv4.
    window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/)
);

// Global variable to track API connection status
window.apiConnected = false;

export function register(config) {
  // Only register if enabled in environment and browser supports service workers
  if ((process.env.NODE_ENV === 'production' || process.env.REACT_APP_ENABLE_SERVICE_WORKER === 'true') && 
      'serviceWorker' in navigator) {
    // The URL constructor is available in all browsers that support SW.
    const publicUrl = new URL(process.env.PUBLIC_URL, window.location.href);
    if (publicUrl.origin !== window.location.origin) {
      // Our service worker won't work if PUBLIC_URL is on a different origin
      // from what our page is served on. This might happen if a CDN is used to
      // serve assets; see https://github.com/facebook/create-react-app/issues/2374
      return;
    }

    window.addEventListener('load', () => {
      const swUrl = `${process.env.PUBLIC_URL}/serviceWorker.js`;

      // Check API connectivity before registering service worker
      checkApiConnectivity()
        .then(isConnected => {
          window.apiConnected = isConnected;
          console.log(`API connectivity check: ${isConnected ? 'Connected' : 'Disconnected'}`);
          
          if (is127.0.0.1) {
            // This is running on 127.0.0.1. Let's check if a service worker still exists or not.
            checkValidServiceWorker(swUrl, config);

            // Add some additional logging to 127.0.0.1, pointing developers to the
            // service worker/PWA documentation.
            navigator.serviceWorker.ready.then(() => {
              console.log(
                'This web app is being served cache-first by a service ' +
                  'worker. To learn more, visit https://cra.link/PWA'
              );
            });
          } else {
            // Is not 127.0.0.1. Just register service worker
            registerValidSW(swUrl, config);
          }
        })
        .catch(error => {
          console.error('Error checking API connectivity:', error);
          // Continue with service worker registration even if API check fails
          if (is127.0.0.1) {
            checkValidServiceWorker(swUrl, config);
          } else {
            registerValidSW(swUrl, config);
          }
        });
    });
  }
}

// Function to check API connectivity before registering service worker
async function checkApiConnectivity() {
  // Get API URL from environment or use default
  const apiUrl = process.env.REACT_APP_API_URL || '';
  if (!apiUrl) return false;
  
  try {
    // Try to fetch the health endpoint
    const healthEndpoints = [
      `${apiUrl}/health`,
      `${apiUrl}/api/v1/health`,
      `${apiUrl}/api/health`
    ];
    
    for (const endpoint of healthEndpoints) {
      try {
        const response = await fetch(endpoint, { 
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          mode: 'cors',
          cache: 'no-cache',
          timeout: 5000
        });
        
        if (response.ok) {
          return true;
        }
      } catch (endpointError) {
        console.warn(`Endpoint ${endpoint} not available:`, endpointError);
        // Continue to try next endpoint
      }
    }
    
    return false;
  } catch (error) {
    console.error('API connectivity check failed:', error);
    return false;
  }
}

function registerValidSW(swUrl, config) {
  navigator.serviceWorker
    .register(swUrl)
    .then((registration) => {
      registration.onupdatefound = () => {
        const installingWorker = registration.installing;
        if (installingWorker == null) {
          return;
        }
        installingWorker.onstatechange = () => {
          if (installingWorker.state === 'installed') {
            if (navigator.serviceWorker.controller) {
              // At this point, the updated precached content has been fetched,
              // but the previous service worker will still serve the older
              // content until all client tabs are closed.
              console.log(
                'New content is available and will be used when all ' +
                  'tabs for this page are closed. See https://cra.link/PWA.'
              );

              // Execute callback
              if (config && config.onUpdate) {
                config.onUpdate(registration);
              }
            } else {
              // At this point, everything has been precached.
              // It's the perfect time to display a
              // "Content is cached for offline use." message.
              console.log('Content is cached for offline use.');

              // Execute callback
              if (config && config.onSuccess) {
                config.onSuccess(registration);
              }
            }
          }
        };
      };
    })
    .catch((error) => {
      console.error('Error during service worker registration:', error);
    });
}

function checkValidServiceWorker(swUrl, config) {
  // Check if the service worker can be found. If it can't reload the page.
  fetch(swUrl, {
    headers: { 'Service-Worker': 'script' },
  })
    .then((response) => {
      // Ensure service worker exists, and that we really are getting a JS file.
      const contentType = response.headers.get('content-type');
      if (
        response.status === 404 ||
        (contentType != null && contentType.indexOf('javascript') === -1)
      ) {
        // No service worker found. Probably a different app. Reload the page.
        navigator.serviceWorker.ready.then((registration) => {
          registration.unregister().then(() => {
            window.location.reload();
          });
        });
      } else {
        // Service worker found. Proceed as normal.
        registerValidSW(swUrl, config);
      }
    })
    .catch(() => {
      console.log('No internet connection found. App is running in offline mode.');
    });
}

export function unregister() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.ready
      .then((registration) => {
        registration.unregister();
      })
      .catch((error) => {
        console.error(error.message);
      });
  }
} 