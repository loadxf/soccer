// Fix for duplicate SportsIcon issue
// This file scans imports in Layout.js and De,o.js and logs any duplicates

const fs = require('fs');
const path = require('path');

console.log('Checking for duplicate imports...');

// Files to check
const files = [
  path.join(__dirname, 'src/frontend/src/components/Layout.js'),
  path.join(__dirname, 'src/frontend/src/pages/Demo.js')
];

// Function to replace SportsIcon in Layout.js with SoccerIcon
function fixLayoutFile() {
  const layoutPath = path.join(__dirname, 'src/frontend/src/components/Layout.js');
  
  try {
    let content = fs.readFileSync(layoutPath, 'utf8');
    
    // Replace the import
    content = content.replace(
      /SportsSoccer as SportsIcon/g, 
      'SportsSoccer as SoccerIcon'
    );
    
    // Replace all usage instances
    content = content.replace(
      /<SportsIcon /g,
      '<SoccerIcon '
    );
    
    // Write the file back
    fs.writeFileSync(layoutPath, content, 'utf8');
    console.log('✅ Successfully fixed Layout.js');
    return true;
  } catch (error) {
    console.error('Error fixing Layout.js:', error);
    return false;
  }
}

// Function to fix serviceWorkerRegistration.js
function fixServiceWorkerFile() {
  const swPath = path.join(__dirname, 'src/frontend/src/serviceWorkerRegistration.js');
  
  try {
    let content = fs.readFileSync(swPath, 'utf8');
    
    // Check if there's a const declaration without initialization
    if (content.includes('const serviceWorkerConfig;')) {
      // Add initialization
      content = content.replace(
        'const serviceWorkerConfig;',
        'const serviceWorkerConfig = {};'
      );
      
      // Write the file back
      fs.writeFileSync(swPath, content, 'utf8');
      console.log('✅ Successfully fixed serviceWorkerRegistration.js');
      return true;
    } else {
      console.log('⚠️ No uninitialized const found in serviceWorkerRegistration.js');
    }
    
    return true;
  } catch (error) {
    console.error('Error fixing serviceWorkerRegistration.js:', error);
    return false;
  }
}

// Fix Layout.js
const layoutFixed = fixLayoutFile();

// Fix serviceWorkerRegistration.js
const serviceWorkerFixed = fixServiceWorkerFile();

if (layoutFixed && serviceWorkerFixed) {
  console.log('✅ All fixes applied successfully');
} else {
  console.error('⚠️ Some fixes were not applied. Check logs above.');
} 