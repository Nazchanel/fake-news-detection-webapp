import requests

url = "http://127.0.0.1:5000/output"
input = "dsad"

data = {
    "test": input
}
s = requests.session()
response = s.post(url=url, data=data)

print(response.text)