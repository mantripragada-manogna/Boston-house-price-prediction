import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'RM':5.627, 'LSTAT':22.88, 'DIS':1.8172, 'CRIM':9.39063, 'PTRATIO': 20.2, 'TAX': 666.})

print(r.json())
