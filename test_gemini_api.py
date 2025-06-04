import requests
import json

def test_gemini_api(api_key):
    """Test if Gemini API is working"""
    
    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
    
    # Simple test prompt
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Say 'Hello, Gemini is working!' if you can read this."
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 100,
            "temperature": 0.7,
        }
    }
    
    try:
        print("Testing Gemini API...")
        response = requests.post(url, headers=headers, json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Gemini API is working.")
            
            # Extract the response text
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print(f"Gemini says: {text}")
            else:
                print("Response structure:", json.dumps(result, indent=2))
        else:
            print("❌ ERROR! Gemini API failed.")
            print(f"Error response: {response.text}")
            
            # Common error explanations
            if response.status_code == 400:
                print("\nPossible issues:")
                print("- Invalid API key format")
                print("- API key has incorrect permissions")
            elif response.status_code == 403:
                print("\nPossible issues:")
                print("- API key is invalid or revoked")
                print("- API not enabled for your project")
                print("- Billing not set up (if required)")
            elif response.status_code == 429:
                print("\nRate limit exceeded. Wait a bit and try again.")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\nCheck your internet connection and try again.")

if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = input("Enter your Gemini API key: ")
    
    if not API_KEY or API_KEY == "your-api-key-here":
        print("Please enter a valid API key!")
    else:
        test_gemini_api(API_KEY)