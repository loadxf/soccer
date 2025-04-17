/**
 * Console Warning Suppressor for Progressive Web Apps
 * 
 * This script suppresses common browser console warnings in React applications:
 * - Feature policy warnings (ambient-light-sensor, battery, document-domain, etc.)
 * - Sandbox iframe warnings
 * - Content Security Policy warnings
 * - Other non-critical browser warnings
 * 
 * Place this file in the public directory and it will be loaded early.
 */

(function() {
  // Check if already applied to avoid duplicate applications
  if (window.__consoleSuppressorApplied) {
    return;
  }
  
  // Override console.warn to filter warnings
  const originalWarn = console.warn;
  console.warn = function(...args) {
    // Only filter string warnings
    if (args.length > 0 && typeof args[0] === 'string') {
      // Feature policy warnings
      if (args[0].includes('feature policy') || 
          args[0].includes('Unrecognized feature') ||
          args[0].match(/feature:.*is not recognized/i) ||
          
          // Specific feature policy warnings
          args[0].includes('ambient-light-sensor') ||
          args[0].includes('battery') ||
          args[0].includes('document-domain') ||
          args[0].includes('layout-animations') ||
          args[0].includes('legacy-image-formats') ||
          args[0].includes('oversized-images') ||
          args[0].includes('vr') ||
          args[0].includes('wake-lock') ||
          
          // Misc browser warnings
          args[0].includes('The Content Security Policy directive')) {
        // Suppress these warnings
        return;
      }
      
      // Sandboxed iframe warnings
      if (args[0].includes('sandbox attribute') || 
          args[0].includes('allow-scripts') || 
          args[0].includes('allow-same-origin') ||
          args[0].includes('can escape its sandboxing') ||
          args[0].includes('An iframe which has both')) {
        // Suppress these warnings  
        return;
      }
    }
    
    // Pass through all other warnings
    originalWarn.apply(console, args);
  };
  
  // Override console.error to filter certain errors
  const originalError = console.error;
  console.error = function(...args) {
    // Only filter string errors
    if (args.length > 0 && typeof args[0] === 'string') {
      // Iframe sandbox errors
      if (args[0].includes('sandbox') || 
          args[0].includes('iframe') || 
          args[0].includes('allow-scripts') || 
          args[0].includes('allow-same-origin')) {
        // Suppress these errors
        return;
      }
      
      // Content security policy errors
      if (args[0].includes('Content Security Policy') ||
          args[0].includes('frame-ancestors')) {
        // Suppress these errors
        return;
      }
    }
    
    // Pass through all other errors
    originalError.apply(console, args);
  };
  
  // Mark as applied
  window.__consoleSuppressorApplied = true;
  
  // Log once that suppression is active (this will be hidden)
  console.log('Browser console warning suppression active');
})(); 