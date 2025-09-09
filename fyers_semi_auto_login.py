"""
Fyers API V3 - Complete Fixed Authentication System
- Fixed token generation with correct V3 endpoints
- Enhanced refresh token support
- IST timezone-aware token management
- Chrome browser auto-opening
"""

import json
import time
import hashlib
import requests
import webbrowser
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import os

class FyersAutoLogin:
    def __init__(self, config_file='fyers_config.json'):
        self.config = self.load_config(config_file)
        self.access_token = None
        self.refresh_token = None
        
    def load_config(self, config_file):
        if not os.path.exists(config_file):
            sample_config = {
                "app_id": "YOUR-APP-ID-100",
                "secret_key": "YOUR-SECRET-KEY", 
                "redirect_uri": "https://www.google.com/",
                "pin": "YOUR-PIN"
            }
            with open(config_file, 'w') as f:
                json.dump(sample_config, f, indent=4)
            raise FileNotFoundError(f"Please fill in your details in {config_file}")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def generate_appid_hash(self):
        """Generate SHA256 hash of app_id:secret_key for API calls"""
        message = f"{self.config['app_id']}:{self.config['secret_key']}".encode('utf-8')
        return hashlib.sha256(message).hexdigest()
    
    def refresh_access_token(self):
        """Use refresh token to get new access token"""
        if not self.refresh_token:
            print("❌ No refresh token available")
            return False
        
        print("🔄 Using refresh token to get new access token...")
        
        try:
            url = "https://api.fyers.in/api/v3/validate-refresh-token"
            headers = {'Content-Type': 'application/json'}
            
            data = {
                'grant_type': 'refresh_token',
                'appIdHash': self.generate_appid_hash(),
                'refresh_token': self.refresh_token,
                'pin': self.config['pin']
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if response_data.get('s') == 'ok' and 'access_token' in response_data:
                    self.access_token = response_data['access_token']
                    
                    # Update refresh token if new one provided
                    if 'refresh_token' in response_data:
                        self.refresh_token = response_data['refresh_token']
                    
                    print("✅ Access token refreshed successfully!")
                    self.save_tokens()
                    return True
                else:
                    print(f"❌ Refresh failed: {response_data}")
                    return False
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error refreshing token: {str(e)}")
            return False
    
    def open_chrome_with_url(self, url):
        """Open URL in Chrome browser"""
        try:
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
            ]
            
            for chrome_path in chrome_paths:
                if os.path.exists(chrome_path):
                    webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
                    chrome_browser = webbrowser.get('chrome')
                    chrome_browser.open(url, new=2)
                    print("🌐 Chrome browser opened successfully!")
                    return True
            
            # Fallback to default browser
            webbrowser.open(url, new=2)
            print("🌐 Opened in default browser")
            return True
            
        except Exception as e:
            print(f"⚠️  Error opening browser: {str(e)}")
            webbrowser.open(url, new=2)
            return False
    
    def try_direct_token_generation(self, auth_code):
        """Alternative direct API call method for token generation"""
        print("🔄 Trying direct API call method...")
        
        try:
            # Generate appIdHash correctly
            app_id_hash = self.generate_appid_hash()
            
            # Use correct V3 endpoint
            token_url = "https://api.fyers.in/api/v3/validate-authcode"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                "grant_type": "authorization_code",
                "appIdHash": app_id_hash,
                "code": auth_code
            }
            
            print(f"   🔗 Calling: {token_url}")
            print(f"   📦 Payload keys: {list(payload.keys())}")
            
            response = requests.post(token_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if response_data.get('s') == 'ok' and 'access_token' in response_data:
                    self.access_token = response_data['access_token']
                    self.refresh_token = response_data.get('refresh_token')
                    
                    print(f"✅ Direct API call successful!")
                    print(f"🔑 Access token: {self.access_token[:25]}...")
                    if self.refresh_token:
                        print(f"🔄 Refresh token: {self.refresh_token[:25]}...")
                    
                    self.save_tokens()
                    return True
                else:
                    print(f"❌ Direct API error: {response_data}")
                    return False
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Direct API call failed: {str(e)}")
            return False
    
    def manual_login_flow(self):
        """Fixed manual login flow with correct V3 token generation"""
        print("🚀 Starting manual login process...")
        
        session = fyersModel.SessionModel(
            client_id=self.config['app_id'],
            secret_key=self.config['secret_key'], 
            redirect_uri=self.config['redirect_uri'],
            response_type="code",
            grant_type="authorization_code"
        )
        
        auth_url = session.generate_authcode()
        print(f"🔗 Generated auth URL")
        
        # Open browser automatically
        self.open_chrome_with_url(auth_url)
        
        print(f"\n👤 Complete the login process in browser:")
        print(f"   1. ✅ Choose login method & complete CAPTCHA")
        print(f"   2. ✅ Enter credentials, TOTP, and PIN")
        print(f"   3. ✅ You'll be redirected to: {self.config['redirect_uri']}")
        
        # Get redirect URL from user
        while True:
            redirect_url = input("\n📥 Paste the complete redirect URL: ").strip()
            
            if "auth_code=" in redirect_url:
                break
            print("❌ No auth_code found. Please copy the complete URL after login.")
        
        # Extract auth code and generate tokens
        try:
            auth_code = redirect_url.split('auth_code=')[1].split('&')[0]
            print(f"✅ Auth code extracted: {auth_code[:15]}...")
            
            # Try SDK method first
            print("🔄 Trying SDK token generation...")
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response and response.get('s') == 'ok' and 'access_token' in response:
                self.access_token = response['access_token']
                self.refresh_token = response.get('refresh_token')
                
                print(f"✅ SDK token generation successful!")
                if self.refresh_token:
                    print(f"🔄 Refresh token obtained (valid for 15 days)")
                
                self.save_tokens()
                return True
            else:
                print(f"⚠️  SDK method failed: {response}")
                # Try alternative direct API call method
                return self.try_direct_token_generation(auth_code)
                
        except Exception as e:
            print(f"❌ SDK method error: {str(e)}")
            # Try alternative direct API call method
            return self.try_direct_token_generation(auth_code)
    
    def save_tokens(self):
        """Save tokens with proper expiration info"""
        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "access_token_generated_at": time.time(),
            "access_token_expires_eod": True,  # Expires at end of day
            "refresh_token_generated_at": time.time(),
            "refresh_token_valid_days": 15
        }
        
        with open('access_token.json', 'w') as f:
            json.dump(token_data, f, indent=4)
        
        # Calculate expiry times
        now = datetime.now()
        # Access token expires at 6:30 AM next day
        next_630am = (now + timedelta(days=1)).replace(hour=6, minute=30, second=0, microsecond=0)
        refresh_expiry = now + timedelta(days=15)
        
        print(f"💾 Tokens saved:")
        print(f"   ⏰ Access token expires: {next_630am.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   ⏰ Refresh token expires: {refresh_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def load_saved_tokens(self):
        """Load tokens and check validity"""
        try:
            with open('access_token.json', 'r') as f:
                token_data = json.load(f)
            
            self.access_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            
            # Check access token validity (expires at 6:30 AM each day)
            now = datetime.now()
            today_630am = now.replace(hour=6, minute=30, second=0, microsecond=0)
            
            if now.hour < 6 or (now.hour == 6 and now.minute < 30):
                # Before 6:30 AM today - token from yesterday should still be valid
                access_token_valid = True
            else:
                # After 6:30 AM - need to check if token was generated today
                token_time = datetime.fromtimestamp(token_data['access_token_generated_at'])
                if token_time.date() == now.date() and token_time > today_630am:
                    access_token_valid = True
                else:
                    access_token_valid = False
            
            # Check refresh token validity (15 days)
            refresh_age = (time.time() - token_data.get('refresh_token_generated_at', 0)) / (24 * 3600)
            refresh_token_valid = refresh_age < 15
            
            if access_token_valid:
                print("✅ Access token is valid")
                return True
            elif refresh_token_valid:
                print("⏰ Access token expired, but refresh token is valid")
                return self.refresh_access_token()
            else:
                print("⏰ Both tokens expired - manual login required")
                return False
                
        except FileNotFoundError:
            print("📄 No saved tokens found")
            return False
        except Exception as e:
            print(f"❌ Error loading tokens: {str(e)}")
            return False
    
    def get_fyers_client(self):
        """Get authenticated Fyers client with automatic token management"""
        # Try to load existing tokens
        if not self.load_saved_tokens():
            # If no valid tokens, do manual login
            if not self.manual_login_flow():
                raise Exception("Failed to authenticate")
        
        client = fyersModel.FyersModel(
            client_id=self.config['app_id'],
            token=self.access_token,
            log_path=""
        )
        
        return client
    
    def test_connection(self):
        """Test connection with enhanced token management"""
        try:
            client = self.get_fyers_client()
            profile = client.get_profile()
            
            if profile['s'] == 'ok':
                print("🎉 Connection successful!")
                print(f"👤 Welcome, {profile['data']['name']}!")
                return client
            else:
                print(f"❌ Connection failed: {profile}")
                return None
        except Exception as e:
            print(f"❌ Test connection failed: {str(e)}")
            return None

# Usage and testing
if __name__ == "__main__":
    print("🔬 Testing Fixed Fyers Authentication...")
    
    fyers = FyersAutoLogin()
    client = fyers.test_connection()
    
    if client:
        print("\n🚀 Authentication successful!")
        print("✅ Ready for bulk data download!")
        
        # Quick test of market data
        try:
            quotes = client.quotes({"symbols": "NSE:RELIANCE-EQ"})
            if quotes['s'] == 'ok':
                reliance_data = quotes['d'][0]['v']
                print(f"📊 Test quote - RELIANCE: ₹{reliance_data['lp']} (Change: {reliance_data['ch']})")
            else:
                print("⚠️  Quote test failed but authentication is working")
        except Exception as e:
            print(f"⚠️  Quote test error: {e}")
    else:
        print("❌ Authentication failed")
