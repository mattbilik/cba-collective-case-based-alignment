import subprocess
import json
import requests

config_file = "aws_config.json"
with open(config_file) as f:
    config = json.load(f)
AWS_REGION = config["aws_region"]
TOKEN = config["aws_token"]

def query_bedrock(prompt, model="us.anthropic.claude-sonnet-4-6"):
    url = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/model/{model}/converse"
    response = requests.post(
      url,
      headers={
          "Content-Type": "application/json",
          "Authorization": f"Bearer {TOKEN}"
      },
      json={"messages": [{"role": "user", "content": [{"text": prompt}]}]}
    ).content.decode('utf-8')
    response = json.loads(response)["output"]["message"]["content"]
    try:
        if "kimi" in model:
            if len(response) == 1:
                message = ""
            else:
                message = response[1]["text"]
        else:
            message = response[0]["text"]
    except IndexError:
        print(response)
        message = ""
    return message
