import requests
import time

#Render Server URL
URL = "https://weather-dashboard-6wze.onrender.com/"

print("Starting Keep-Alive script...")

while True:
    try:
        response = requests.get(URL)
        print(f"Ping successful! Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error pinging server: {e}")
    
    # send ping request every 300 seconds
    time.sleep(300)