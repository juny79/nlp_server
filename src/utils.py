import os
import time
import requests

def load_api_key(env_name: str):
    key = os.getenv(env_name)
    if key is None:
        raise RuntimeError(f"환경 변수 {env_name} 에 API 키가 없습니다.")
    return key

def call_solar_api(base_url, api_key, model, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    if response.status_code != 200:
        print("API ERROR:", response.text)
        return ""  # API 에러 시 빈 문자열 반환
    return response.json()["choices"][0]["message"]["content"]
