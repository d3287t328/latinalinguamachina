import requests

def get_chatgpt_response(prompt):
    api_key = "your_api_key"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "text-davinci-002",
        "prompt": prompt,
        "max_tokens": 100
    }
    response = requests.post("https://api.openai.com/v1/engines/davinci-codex/completions", headers=headers, json=data)
    response_json = response.json()
    return response_json["choices"][0]["text"].strip()
