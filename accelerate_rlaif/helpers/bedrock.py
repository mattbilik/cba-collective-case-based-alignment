import subprocess
import json

def query_bedrock(prompt, model="us.anthropic.claude-sonnet-4-6"):
    messages = [{'role': 'user', 'content': [{'text': prompt}]}]
    cmd = [
        'aws', 'bedrock-runtime', 'converse',
        '--region',   AWS_REGION,
        '--model-id', model,
        '--messages', json.dumps(messages),
        '--system',   json.dumps([{'text': ""}]),
        '--output',   'json',
    ]
    if AWS_PROFILE:
        cmd += ['--profile', AWS_PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)['output']['message']['content'][0]['text']
