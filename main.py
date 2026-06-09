import datetime
import time
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
    full = response.json()
    if "error" in full:
        raise RuntimeError(f"OpenRouter error {full['error']['code']}: {full['error']['message']}")
    request_result = {
            "response": full['choices'][0]['message']['content'],
            "prompt_tokens": full['usage']['prompt_tokens'],
            "completion_tokens": full['usage']['completion_tokens'],
            "cost": full['usage']['cost'],
        }
    return request_result

def exact_match(response, expected):
    return response.strip().lower() == expected.strip().lower()

def save_results(config, results, timestamp):
    benchmark_name =  config['name']
    run_id = f"{timestamp}_{benchmark_name}"
    data = {
        "run_id": run_id,
        "benchmark": config['name'],
        "judge_model": config['judge_model'] if 'judge_model' in config else None,
        "temperature": config['inference']['temperature'] if 'temperature' in config['inference'] else 0,
        "max_tokens": config['inference']['max_tokens'] if 'max_tokens' in config['inference'] else 4096,
        "quality_threshold": config['quality_threshold'] if 'quality_threshold' in config else None,
        "timestamp": timestamp,
        "results": results
    }
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{run_id}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return file_path


if __name__ == "__main__":
    load_dotenv()
    filepath = 'benchmarks/first_classification.yaml'
    config = load_config(filepath)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results = []
    test_case_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0

    for model in config['models']:
        print(f"Model: {model}\n")
        for prompt in config['prompts']:
            success = 0
            total = 0
            test_case_results = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0    
            print(f"Prompt: {prompt['id']}")
            for test_case in config['test_cases']:
                print(f"Test case: {test_case['id']}")    
                rendered = render_template(prompt, config['variables'], test_case)
                print(f"Rendered prompt: {rendered}")
                response = openrouter_request(rendered, model, config['inference']['temperature'], config['inference']['max_tokens'])
                time.sleep(5)
                exact_match_result = exact_match(response['response'], test_case['expected_output'])
                print(f"Response: {response['response']} Prompt tokens: {response['prompt_tokens']} Completion tokens: {response['completion_tokens']} Cost: ${response['cost']:.6f}")
                prompt_tokens = response['prompt_tokens']
                completion_tokens = response['completion_tokens']
                cost = response['cost']
                test_case_results.append({
                    "id": test_case['id'],
                    "tags": test_case['tags'],
                    "evaluation": test_case['evaluation'],
                    "match_result": exact_match_result,
                    "response": response['response'],
                    "expected": test_case['expected_output'],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": cost
                })
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += cost


                print(f"Expected: {test_case['expected_output']}")
                print(f"Exact match: {exact_match_result}\n")
                if exact_match_result:
                    success += 1
                total += 1

            score = success / total * 100
            results.append({
                "model": model,
                "prompt_id": prompt['id'],
                "success": success,
                "total": total,
                "score": score,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_cost": total_cost,
                "test_cases": test_case_results
            })
            print(f"Quality score: {success}/{total} ({score:.1f}%)")
            print(f"Total prompt tokens: {total_prompt_tokens}")
            print(f"Total completion tokens: {total_completion_tokens}")
            print(f"Total cost: ${total_cost:.6f}")
    file = save_results(config, results, timestamp)
    print(f"Results saved to {file}")