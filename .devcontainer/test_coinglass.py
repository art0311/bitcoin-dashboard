import requests

API_KEY = "65c076928f664c6b9c5d67f21d83aac0"

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/flow-history"

headers = {
    "accept": "application/json",
    "CG-API-KEY": API_KEY,
}

r = requests.get(url, headers=headers, timeout=15)

print("Status:", r.status_code)
print(r.text)
