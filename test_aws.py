import requests

url = "http://<IP SERWERA>:3000/predict" 
data = {"smiles": "CCO"}

try:
    response = requests.post(url, json=data)
    print("Status:", response.status_code)
    print("Wynik z chmury:", response.json())
except Exception as e:
    print("Błąd połączenia:", e)