import requests
from config import api_key
import json


base_url = 'https://www.alphavantage.co'
symbol = 'BTc'
url = f'{base_url}/query?function=CRYPTO_INTRADAY&symbol={symbol}&market=CAD&interval=5min&apikey={api_key}'

res = requests.get(url)
data = json.loads(res.content)
print(data)
with open('data.json', 'w')as f:
    json.dump(data, f)
