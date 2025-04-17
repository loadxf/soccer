/**
 * Browser Console Warning Suppressor
 * 
 * This script suppresses common browser console warnings that appear in Streamlit applications
 * - Feature policy warnings (ambient-light-sensor, battery, etc.)
 * - Sandbox iframe warnings 
 * - Other non-critical browser warnings
 * 
 * Include this in your Streamlit app with:
 * st.components.v1.html(open("browser_console_suppressor.js").read(), height=0)
 */

// Self-executing function to avoid polluting global namespace
(function() {
    // Track if we've already applied suppression
    if (window.__warningsSuppressed) return;
    
    // Override console.warn to filter out specific warnings
    const originalWarn = console.warn;
    console.warn = function(...args) {
        // Filter out feature policy warnings
        if (args.length > 0 && typeof args[0] === 'string') {
            // Skip warnings about feature policy
            if (args[0].includes('feature policy') || 
                args[0].includes('Unrecognized feature') || 
                args[0].includes('ambient-light-sensor') ||
                args[0].includes('battery') ||
                args[0].includes('document-domain') ||
                args[0].includes('layout-animations') ||
                args[0].includes('legacy-image-formats') ||
                args[0].includes('oversized-images') ||
                args[0].includes('vr') ||
                args[0].includes('wake-lock') ||
                args[0].includes('browser console warning')) {
                return; // Suppress these warnings
            }
            
            // Skip warnings about sandbox iframes
            if (args[0].includes('sandbox attribute') || 
                args[0].includes('allow-scripts') || 
                args[0].includes('allow-same-origin') ||
                args[0].includes('can escape its sandboxing')) {
                return; // Suppress these warnings
            }
        }
        
        // Pass through other warnings
        originalWarn.apply(console, args);
    };
    
    // Also override console.error to catch some error messages from sandboxed iframes
    const originalError = console.error;
    console.error = function(...args) {
        // Filter out iframe sandbox errors
        if (args.length > 0 && typeof args[0] === 'string') {
            if (args[0].includes('sandbox') || 
                args[0].includes('iframe') || 
                args[0].includes('allow-scripts') || 
                args[0].includes('allow-same-origin')) {
                return; // Suppress these errors
            }
        }
        
        // Pass through other errors
        originalError.apply(console, args);
    };
    
    // Add Content-Security-Policy meta tag to suppress other warnings
    const addCSPMetaTag = function() {
        // Don't add if already exists
        if (document.querySelector('meta[http-equiv="Content-Security-Policy"]')) {
            return;
        }
        
        // Create meta tag
        const meta = document.createElement('meta');
        meta.httpEquiv = 'Content-Security-Policy';
        meta.content = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:; frame-ancestors 'self';";
        document.head.appendChild(meta);
    };
    
    // Try to add CSP meta tag if document is ready
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        addCSPMetaTag();
    } else {
        document.addEventListener('DOMContentLoaded', addCSPMetaTag);
    }
    
    // Mark as applied
    window.__warningsSuppressed = true;
    
    // Optional: Log once that suppression is active
    console.log('Browser console warning suppression active');
})(); 