import yaml
from dotenv import load_dotenv
import os
import httpx
import json
from jinja2 import Template

def load_config(filepath):
    with open(filepath) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return config

def render_template(prompt, variables, test_case):
    template = Template(prompt['template'])
    if variables:
        render_kwargs = dict(variables)
    else:
        render_kwargs = {}
    if test_case['variables']:
        render_kwargs.update(test_case['variables'])
    rendered_prompt = template.render(**render_kwargs)
    return rendered_prompt

def openrouter_request(prompt, model, temperature=0, max_tokens=4096):
    api_key = os.getenv("OPENROUTER_API_KEY")
    response = httpx.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "localhost", # Optional. Site URL for rankings on openrouter.ai.
        "X-OpenRouter-Title": "evalblink", # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps({
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ]
    })
    )
    response = response.json()
    response = response['choices'][0]['message']['content']
    return(response)

def exact_match(response, expected):
    return response.strip().lower() == expected.strip().lower()

if __name__ == "__main__":
    load_dotenv()
    filepath = 'benchmarks/first_classification.yaml'
    config = load_config(filepath)
    
    for model in config['models']:
        print(f"Model: {model}\n")
        for prompt in config['prompts']:
            success = 0
            total = 0
            print(f"Prompt: {prompt['id']}")
            for test_case in config['test_cases']:
                print(f"Test case: {test_case['id']}")    
                rendered = render_template(prompt, config['variables'], test_case)
                print(f"Rendered prompt: {rendered}")
                response = openrouter_request(rendered, model, **config['inference'])
                exact_match_result = exact_match(response, test_case['expected_output'])
                print(f"Response: {response}")
                print(f"Expected: {test_case['expected_output']}")
                print(f"Exact match: {exact_match_result}\n")
                if exact_match_result:
                    success += 1
                total += 1
            score = success / total * 100
            print(f"Quality score: {success}/{total} ({score:.1f}%)")