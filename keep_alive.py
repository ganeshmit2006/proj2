import time
import urllib.request

URL = "https://proj2-u4hn.onrender.com/api/"

while True:
    try:
        with urllib.request.urlopen(URL) as response:
            print(f"Pinged {URL}, status: {response.status}")
    except Exception as e:
        print(f"Error pinging {URL}: {e}")
    time.sleep(120)  # Wait 2 minutes
