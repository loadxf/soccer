import requests
import json
import sys
import webbrowser
import time
import os
from pathlib import Path

def log_headers_and_response(response):
    """Log detailed information about a response"""
    print(f"\nRequest URL: {response.url}")
    print(f"Status Code: {response.status_code}")
    print("Response Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    
    print("\nResponse Content (first 500 chars):")
    try:
        if 'application/json' in response.headers.get('Content-Type', ''):
            print(json.dumps(response.json(), indent=2))
        else:
            print(response.text[:500] + '...' if len(response.text) > 500 else response.text)
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(response.text[:500] + '...' if len(response.text) > 500 else response.text)

def test_cors():
    """Test CORS handling by the API"""
    print("\n=== Testing CORS Handling ===")
    try:
        # Test API root with CORS headers that a browser would send
        headers = {
            'Origin': 'http://127.0.0.1:8501',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'http://127.0.0.1:8501/'
        }
        
        response = requests.get('http://127.0.0.1:8080/api/v1/', headers=headers)
        log_headers_and_response(response)
        
        # Check if CORS headers are present
        if 'Access-Control-Allow-Origin' in response.headers:
            print(f"\n✅ API correctly returns CORS headers")
            print(f"Access-Control-Allow-Origin: {response.headers['Access-Control-Allow-Origin']}")
        else:
            print("\n⚠️ API doesn't return CORS headers, which might cause browser issues")
            
    except Exception as e:
        print(f"Error testing CORS: {e}")

def test_browser_simulation():
    """Open test page that makes fetch requests to the API"""
    # Create a small HTML file that makes fetch requests to the API
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .result { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 10px; }
            .success { color: green; }
            .error { color: red; }
            button { margin-top: 10px; padding: 5px 10px; }
            h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>API Connection Test</h1>
        
        <h2>Testing with 127.0.0.1</h2>
        <button onclick="testHealth('127.0.0.1')">Test Health Endpoint</button>
        <button onclick="testRoot('127.0.0.1')">Test Root Endpoint</button>
        <button onclick="testTeams('127.0.0.1')">Test Teams Endpoint</button>
        
        <h2>Testing with localhost</h2>
        <button onclick="testHealth('localhost')">Test Health Endpoint</button>
        <button onclick="testRoot('localhost')">Test Root Endpoint</button>
        <button onclick="testTeams('localhost')">Test Teams Endpoint</button>
        
        <div id="results"></div>

        <script>
            async function testEndpoint(host, endpoint, name) {
                const url = `http://${host}:8080/api/v1/${endpoint}`;
                const resultDiv = document.getElementById('results');
                const endpointResult = document.createElement('div');
                endpointResult.className = 'result';
                endpointResult.innerHTML = `<h3>Testing ${name} with ${host}</h3><p>Connecting to: ${url}</p>`;
                resultDiv.prepend(endpointResult);
                
                try {
                    console.log(`Fetching ${url}...`);
                    const startTime = new Date();
                    const response = await fetch(url);
                    const elapsed = new Date() - startTime;
                    
                    console.log('Response status:', response.status);
                    console.log('Response headers:', response.headers);
                    
                    const data = await response.json();
                    console.log('Response data:', data);
                    
                    endpointResult.innerHTML += `
                        <p class="success">✅ Success! Status: ${response.status} (${elapsed}ms)</p>
                        <p>Response:</p>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    `;
                } catch (error) {
                    console.error('Error:', error);
                    endpointResult.innerHTML += `
                        <p class="error">❌ Error: ${error.message}</p>
                        <p>Check browser console for details (F12)</p>
                    `;
                }
            }
            
            function testHealth(host) {
                testEndpoint(host, 'health', 'Health Endpoint');
            }
            
            function testRoot(host) {
                testEndpoint(host, '', 'Root Endpoint');
            }
            
            function testTeams(host) {
                testEndpoint(host, 'teams', 'Teams Endpoint');
            }
        </script>
    </body>
    </html>
    """
    
    # Write HTML to a temporary file
    test_page_path = Path("api_test.html")
    with open(test_page_path, "w") as f:
        f.write(html_content)
    
    print(f"\n=== Opening Browser Test Page ===")
    print(f"Created test page at: {test_page_path.absolute()}")
    print("Please open this file in your browser to test API connections.")
    print("This will help diagnose if there's a browser-specific issue.")
    
    # Open the page in the default browser
    webbrowser.open(test_page_path.as_uri())

def main():
    print("=== API Browser Connection Diagnostic ===")
    print("This script helps diagnose browser connection issues with the API server.")
    
    # First test regular Python requests
    print("\n=== Testing Direct API Connection ===")
    try:
        response = requests.get('http://127.0.0.1:8080/api/v1/health')
        if response.status_code == 200:
            print(f"✅ API health endpoint is accessible: {response.status_code}")
            print(f"Response: {response.json()}")
        else:
            print(f"⚠️ API health endpoint returned status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error connecting to API health endpoint: {e}")
        sys.exit(1)
    
    # Test CORS handling
    test_cors()
    
    # Create and open browser test page
    test_browser_simulation()
    
    print("\nDiagnostic tests completed.")
    print("If the browser test page shows errors but direct requests succeeded,")
    print("the issue is likely related to CORS or browser security settings.")

if __name__ == "__main__":
    main() 