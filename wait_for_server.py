import requests
import time

url = "http://127.0.0.1:8000/"
print(f"Waiting for {url} to become alive...")

for i in range(60):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"SUCCESS: Server is live! Response: {response.text}")
            break
    except Exception:
        pass
    time.sleep(5)
else:
    print("TIMED OUT: Server did not respond within 5 minutes.")
