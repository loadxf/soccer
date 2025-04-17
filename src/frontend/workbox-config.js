module.exports = {
  globDirectory: 'build/',
  globPatterns: [
    '**/*.{html,js,css,png,jpg,jpeg,gif,svg,ico,json,woff,woff2,eot,ttf,otf}'
  ],
  swDest: 'build/serviceWorker.js',
  swSrc: 'public/serviceWorker.js',
  maximumFileSizeToCacheInBytes: 5 * 1024 * 1024, // 5MB
  dontCacheBustURLsMatching: [
    /^utm_/,
    /^fbclid$/
  ]
}; 