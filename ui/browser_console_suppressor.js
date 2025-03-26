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
                args[0].includes('wake-lock')) {
                return; // Suppress these warnings
            }
            
            // Skip warnings about sandbox iframes
            if (args[0].includes('sandbox attribute') || 
                args[0].includes('allow-scripts') || 
                args[0].includes('allow-same-origin')) {
                return; // Suppress these warnings
            }
        }
        
        // Pass through other warnings
        originalWarn.apply(console, args);
    };
    
    // Mark as applied
    window.__warningsSuppressed = true;
    
    // Optional: Log once that suppression is active
    console.log('Browser console warning suppression active');
})(); 