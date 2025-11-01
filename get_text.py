import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

with open("tiny_shakespeare.txt", "w", encoding = "utf-8") as f:
    f.write(response.text)

    