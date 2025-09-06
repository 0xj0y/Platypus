"""
Updated Fyers Semi-Automated Login with Chrome Browser Opening
Specifically opens Chrome instead of default browser
"""

import json
import time
import webbrowser
import os
import subprocess
from fyers_apiv3 import fyersModel

class FyersSemiAutoLogin:
    def __init__(self, config_file='fyers_config.json'):
        self.config = self.load_config(config_file)
        self.access_token = None
        
    def load_config(self, config_file):
        if not os.path.exists(config_file):
            sample_config = {
                "app_id": "YOUR-APP-ID-100",
                "secret_key": "YOUR-SECRET-KEY", 
                "redirect_uri": "https://www.google.com/"
            }
            with open(config_file, 'w') as f:
                json.dump(sample_config, f, indent=4)
            raise FileNotFoundError(f"Please fill in your details in {config_file}")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def find_chrome_path(self):
        """Find Chrome installation path on Windows"""
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def open_chrome_with_url(self, url):
        """Open URL specifically in Chrome browser"""
        try:
            # Method 1: Try webbrowser with Chrome registration
            chrome_path = self.find_chrome_path()
            
            if chrome_path:
                # Register Chrome with webbrowser
                webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
                
                # Open in Chrome
                chrome_browser = webbrowser.get('chrome')
                chrome_browser.open(url, new=2)  # new=2 opens in new tab
                print("🌐 Chrome browser opened successfully!")
                return True
            else:
                # Method 2: Try using 'google-chrome' (works on some systems)
                try:
                    webbrowser.get('google-chrome').open(url, new=2)
                    print("🌐 Chrome browser opened successfully!")
                    return True
                except:
                    # Method 3: Use subprocess to directly call Chrome
                    if chrome_path:
                        subprocess.Popen([chrome_path, url])
                        print("🌐 Chrome browser opened via subprocess!")
                        return True
                    else:
                        print("❌ Chrome not found. Opening in default browser...")
                        webbrowser.open(url, new=2)
                        return False
                        
        except Exception as e:
            print(f"⚠️  Error opening Chrome: {str(e)}")
            print("🔄 Trying default browser...")
            webbrowser.open(url, new=2)
            return False
    
    def generate_login_url(self):
        """Generate login URL for manual login"""
        session = fyersModel.SessionModel(
            client_id=self.config['app_id'],
            secret_key=self.config['secret_key'], 
            redirect_uri=self.config['redirect_uri'],
            response_type="code",
            grant_type="authorization_code"
        )
        
        auth_url = session.generate_authcode()
        return auth_url, session
    
    def extract_auth_code_from_url(self, redirect_url):
        """Extract auth code from the redirect URL"""
        if "auth_code=" in redirect_url:
            return redirect_url.split('auth_code=')[1].split('&')[0]
        else:
            raise ValueError("No auth_code found in the URL")
    
    def login_flow(self):
        """Semi-automated login flow with Chrome browser opening"""
        print("🚀 Starting semi-automated login process...")
        
        # Generate login URL
        auth_url, session = self.generate_login_url()
        
        print(f"\n📋 STEP 1: Opening Chrome Browser for Login")
        print(f"🔗 Generated URL: {auth_url[:60]}...")
        
        # Try to open Chrome specifically
        chrome_opened = self.open_chrome_with_url(auth_url)
        
        if not chrome_opened:
            print(f"\n⚠️  Could not open Chrome specifically")
            print(f"🌐 URL opened in default browser instead")
        
        print(f"\n👤 Complete the login process in Chrome:")
        print(f"   1. ✅ Choose your login method (Mobile number/Client ID)")
        print(f"   2. ✅ Complete the CAPTCHA ('Verify you are human')")
        print(f"   3. ✅ Enter your credentials (mobile/username + password)")
        print(f"   4. ✅ Complete TOTP (6-digit code)")
        print(f"   5. ✅ Enter your PIN (4-digit)")
        
        print(f"\n🔄 After successful login:")
        print(f"   📍 You'll be redirected to: {self.config['redirect_uri']}")
        print(f"   📋 Copy the ENTIRE URL from Chrome's address bar")
        
        # Get redirect URL from user
        print(f"\n⌛ Waiting for you to complete the login in Chrome...")
        
        while True:
            redirect_url = input("\n📥 Paste the complete redirect URL here: ").strip()
            
            if not redirect_url:
                print("❌ Empty URL. Please paste the redirect URL.")
                continue
                
            if "google.com" not in redirect_url and self.config['redirect_uri'] not in redirect_url:
                print("❌ This doesn't look like the redirect URL. Please check and try again.")
                continue
                
            if "auth_code=" not in redirect_url:
                print("❌ No auth_code found in URL. Please make sure you copied the complete URL after login.")
                continue
                
            break
        
        # Extract auth code and generate token
        try:
            auth_code = self.extract_auth_code_from_url(redirect_url)
            print(f"✅ Auth code extracted: {auth_code[:15]}...")
            
            print("🔄 Generating access token...")
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response and 'access_token' in response:
                self.access_token = response['access_token']
                print(f"🔑 Access token generated successfully!")
                
                self.save_access_token()
                print("💾 Token saved for 24-hour automated usage")
                return self.access_token
            else:
                raise Exception("Failed to generate access token from response")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise
    
    def save_access_token(self):
        """Save access token to file"""
        token_data = {
            "access_token": self.access_token,
            "generated_at": time.time(),
            "expires_in_hours": 24
        }
        
        with open('access_token.json', 'w') as f:
            json.dump(token_data, f, indent=4)
        
        expiry_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(time.time() + 86400))
        print(f"⏰ Token valid until: {expiry_time}")
    
    def load_saved_token(self):
        """Load previously saved token"""
        try:
            with open('access_token.json', 'r') as f:
                token_data = json.load(f)
            
            age_hours = (time.time() - token_data['generated_at']) / 3600
            
            if age_hours < 24:
                self.access_token = token_data['access_token']
                remaining_hours = 24 - age_hours
                print(f"✅ Loaded saved access token")
                print(f"⏰ Valid for {remaining_hours:.1f} more hours")
                return True
            else:
                print(f"⏰ Saved token expired ({age_hours:.1f} hours old)")
                return False
        except FileNotFoundError:
            print("📄 No saved token found - new login required")
            return False
    
    def get_fyers_client(self):
        """Get authenticated Fyers client"""
        if not self.access_token:
            if not self.load_saved_token():
                self.login_flow()
        
        client = fyersModel.FyersModel(
            client_id=self.config['app_id'],
            token=self.access_token,
            log_path=""
        )
        
        return client
    
    def test_connection(self):
        """Test the connection"""
        try:
            client = self.get_fyers_client()
            profile = client.get_profile()
            
            if profile['s'] == 'ok':
                print("\n🎉 Connection successful!")
                print(f"👤 Welcome, {profile['data']['name']}!")
                return client
            else:
                print(f"❌ Connection failed: {profile}")
                return None
        except Exception as e:
            print(f"❌ Test connection failed: {str(e)}")
            return None

# Quick test
if __name__ == "__main__":
    print("🧪 Testing Fyers Auto-Login with Chrome Opening...")
    
    fyers = FyersSemiAutoLogin()
    client = fyers.test_connection()
    
    if client:
        print("\n✅ SUCCESS! Your Fyers API is ready!")
        print("🚀 Chrome opened and login completed!")
    else:
        print("\n❌ Setup incomplete. Please check your configuration.")
