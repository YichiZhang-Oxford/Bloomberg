# -*- coding: utf-8 -*-
# @Time    : 2023-06-24 21:00
# @Author  :
# @Email   :
# @File    : summary_generation.py
# @Software: PyCharm
import requests

url = "https://api.openai.com//v1/chat/completions"
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer sk-proj-nDQxyHaMP1h1KuUGm64UT3BlbkFJq9TxpQBl9EUGAt8GgXWn'
}


def build_prompt(context):
  prompt = (f"""Roles: Character: Summary Generation Model.
            Task: Summarize the following report in no more than 50 words. 
            Report:{context}""")
  return prompt


def chat(text):
  payload = {
    "model": "gpt-3.5-turbo-1106",
    "messages": [{"role": "user", "content": text}],
    "temperature": 0,
    "max_tokens": 1024
  }
  response = requests.post(url, headers=headers, json=payload)
  response_json = response.json()
  output = response_json["choices"][0]["message"]["content"]
  return output

if __name__ == '__main__':
  text = "Please generate a report on the trend of oil in English, not exceeding 100 words."
  reporter = chat(text)
  prompt = build_prompt(reporter)
  summary = chat(prompt)
  print(summary)