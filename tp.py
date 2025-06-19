"""
update time: 2025.01.09
verson: 0.1.125
"""
import json
import re
import time
from datetime import datetime, timedelta
from typing import Set, Optional, List, Dict
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import aiohttp
import requests
# 禁用 SSL 警告
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)
urllib3.disable_warnings()

# 新增导入
import tiktoken
from datetime import datetime

debug = True
# 全局变量
last_request_time = 0  # 上次请求的时间戳
cache_duration = 14400  # 缓存有效期，单位：秒 (4小时)

'''用于存储缓存的模型数据'''
cached_models = {
    "object": "list",
    "data": [],
    "version": "1.0.0",
    "provider": "TP",
    "name": "TP",
    "default_locale": "en-US",
    "status": True,
    "time": 20250619
}



# 全局变量：存储所有模型的统计信息
# 格式：{model_name: {"calls": 调用次数, "fails": 失败次数, "last_fail": 最后失败时间}}
MODEL_STATS: Dict[str, Dict] = {}

def record_call(model_name: str, success: bool = True) -> None:
    """
    记录模型调用情况
    Args:
        model_name: 模型名称
        success: 调用是否成功
    """
    global MODEL_STATS
    if model_name not in MODEL_STATS:
        MODEL_STATS[model_name] = {"calls": 0, "fails": 0, "last_fail": None}

    stats = MODEL_STATS[model_name]
    stats["calls"] += 1
    if not success:
        stats["fails"] += 1
        stats["last_fail"] = datetime.now()



'''基础模型'''
base_model = "gpt-4o-mini"

models = [
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4o-mini-2024-07-18",
    "deepseek-v3",
    "uncensored-r1",
    "deepseek-r1",
    "Image-Generator"
]
data = [
            {"id": "gpt-4o-mini","model":"gpt-4o-mini", "object": "model", "created": 17503119683, "owned_by": "OpenAI","type":"text"},
            {"id": "chatgpt-4o-latest","model":"chatgpt-4o-latest", "object": "model", "created": 17503119683, "owned_by": "OpenAI","type":"text"},
            {"id": "gpt-4o-mini-2024-07-18","model":"gpt-4o-mini-2024-07-18", "object": "model", "created": 17503119683, "owned_by": "OpenAI","type":"text"},
            {"id": "deepseek-v3","model":"deepseek-v3", "object": "model", "created": 17503119683, "owned_by": "OpenAI","type":"text"},
            {"id": "deepseek-r1","model":"deepseek-r1", "object": "model", "created": 17503119683, "owned_by": "OpenAI","type":"text"},
            {"id": "uncensored-r1","model":"uncensored-r1", "object": "model", "created": 17503119683, "owned_by": "TP","type":"text"},
            {"id": "Image-Generator","model":"Image-Generator", "object": "model", "created": 17503119683, "owned_by": "TP","type":"image"}
        ]
def get_models():
    """模型值"""
    models = {
        "object": "list",
        "data": data
    }
    return json.dumps(models)

def get_auto_model(cooldown_seconds: int = 300) -> str:
    """异步获取最优模型"""
    return base_model



def reload_check():
    """
    检查并更新系统状态
    @todo
    """
    pass



def is_chatgpt_format(data):
    """Check if the data is in the expected ChatGPT format"""
    try:
        # If the data is a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return False  # If the string can't be parsed, it's not in the expected format

        # Now check if data is a dictionary and contains the necessary structure
        if isinstance(data, dict):
            # Ensure 'choices' is a list and the first item has a 'message' field
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                if "message" in data["choices"][0]:
                    return True
    except Exception as e:
        print(f"Error checking ChatGPT format: {e}")

    return False


def chat_completion_message(
        user_prompt,
        user_id: str = None,
        session_id: str = None,
        system_prompt="You are a helpful assistant.",
        model: str = None,
        stream=False,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0):
    """未来会增加回话隔离: 单人对话,单次会话"""
    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return chat_completion_messages(messages, user_id=user_id, session_id=session_id,
                                    model=model,  stream=stream, temperature=temperature,
                                    max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty)
def chat_completion_messages(
        messages,
        stream=False,
        model: str = None,
        user_id: str = None,
        session_id: str = None,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0):
    # 确保model有效
    if not model or model == "auto":
        model = get_auto_model()
    
    if debug:
        print(f"校准后的model: {model}")
    

    headers_proxy = {
        'accept': 'application/json, text/event-stream',
        'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
        'content-type': 'application/json',
        'dnt': '1',
        'origin': 'https://chat.typegpt.net',
        'priority': 'u=1, i',
        'referer': 'https://chat.typegpt.net/',
        'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
    }

    data_proxy = {
        "messages": messages,
        "stream": stream,
        "model": model,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "top_p": top_p,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "DuckDuckGoLiteSearch",
                    "description": "a search engine. useful for when you need to answer questions about current events. input should be a search query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "keywords for query."
                            },
                            "s": {
                                "type": "number",
                                "description": "can be `0`"
                            },
                            "o": {
                                "type": "string",
                                "description": "can be `json`"
                            },
                            "api": {
                                "type": "string",
                                "description": "can be `d.js`"
                            },
                            "kl": {
                                "type": "string",
                                "description": "wt-wt, us-en, uk-en, ru-ru, etc. Defaults to `wt-wt`."
                            },
                            "bing_market": {
                                "type": "string",
                                "description": "wt-wt, us-en, uk-en, ru-ru, etc. Defaults to `wt-wt`."
                            }
                        },
                        "required": [
                            "q"
                        ]
                    }
                }
            }
        ]
    }
    if debug:
        print(json.dumps(headers_proxy, indent=4))
        print(json.dumps(data_proxy, indent=4))
    return chat_completion(model=model, headers=headers_proxy, payload=data_proxy,stream=stream)



def parse_response(response_text):
    """
    逐行解析SSE流式响应并提取delta.content字段
    包含多层结构校验，确保安全访问嵌套字段
    返回标准API响应格式
    """
    lines = response_text.split('\n')
    result = ""
    created = None
    object_type = None
    
    for line in lines:
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                # 提取第一个data行的元信息
                if isinstance(data, dict) and not created:
                    created = data.get("created")
                    object_type = data.get("object")
                
                # 安全访问嵌套字段，确保是字典类型
                if isinstance(data, dict):
                    # 检查是否存在choices字段且为列表
                    if "choices" in data and isinstance(data["choices"], list):
                        for choice in data["choices"]:
                            # 检查每个choice是否为字典且包含delta字段
                            if isinstance(choice, dict) and "delta" in choice:
                                delta = choice["delta"]
                                # 确保delta是字典且包含content字段
                                if isinstance(delta, dict) and "content" in delta:
                                    content = delta["content"]
                                    # 确保content是字符串类型
                                    if isinstance(content, str):
                                        result += content
            except json.JSONDecodeError:
                continue
    
    # 计算token数量
    enc = tiktoken.get_encoding("cl100k_base")
    completion_tokens = len(enc.encode(result))
    
    # 组装标准响应数据
    response_data = {
        "id": f"chatcmpl-{datetime.now().timestamp()}",
        "object": object_type or "chat.completion",
        "created": created or int(datetime.now().timestamp()),
        "model": "gpt-4o",  # 可根据需求调整来源
        "usage": {
            "prompt_tokens": 0,  # 需要根据实际prompt内容计算
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens
        },
        "choices": [{
            "message": {
                "role": "assistant",
                "content": result
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }
    
    return response_data

def chat_completion(model, headers, payload,stream):
    """处理用户请求并保留上下文"""
    try:
        url = "https://chat.typegpt.net/api/openai/v1/chat/completions"
        if debug:
            print(f"url: {url}")
        response = requests.post(url=url, headers=headers, json=payload, verify=False, timeout=100)
        response.encoding = 'utf-8'
        response.raise_for_status()
        if response.status_code != 200:
            record_call(model, False)
        else:
            record_call(model, True)

        if debug:
            print(response.status_code)
            print(response.text)
        # return response.json()
        if stream:
            if debug:
                print('this is streaming')
            return parse_response(response.text)
        return response.text
    except requests.exceptions.RequestException as e:
        record_call(model, False)
        print(f"请求失败: {e}")
        return "请求失败，请检查网络或参数配置。"
    except (KeyError, IndexError) as e:
        record_call(model, False)
        print(f"解析响应时出错: {e}")
        return "解析响应内容失败。"
    record_call(model, False)
    return {}


if __name__ == '__main__':
    # get_from_js_v3()
    # print("get_models: ", get_models())
    # print("cached_models:", cached_models)
    # print("base_url: ", base_url)
    # print("MODEL_STATS:", MODEL_STATS)
    # print("base_model:",base_model)
    # base_model = "QwQ-32B"
    result = chat_completion_message(user_prompt="你是什么模型？", model=base_model,stream=True)
    print(result)

    # base_model="Llama-4-Scout-Instruct"
    # result = chat_completion_message(user_prompt="你是什么模型？", model=base_model, stream=False)
    # print(result)

    # # 单次对话
    # result1 = chat_completion_message(
    #     user_prompt="你好，请介绍下你自己",
    #     # model=base_model,
    #     temperature=0.3
    # )
    # print(result1)

    # # 多轮对话
    # messages = [
    #     {"role": "system", "content": "你是一个助手"},
    #     {"role": "user", "content": "你好"}
    # ]
    # result2 = chat_completion_messages(messages)
    # print(result2)

 #    msg="""
 #    json 格式化
 # {"object": "list", "data": [{"id": "Qwen2.5-VL-72B-Instruct", "object": "model", "model": "Qwen2.5-VL-72B-Instruct", "created": 1744090984000, "owned_by": "Qwen2.5", "name": "Qwen o1", "description": "Deep thinking,mathematical and writing abilities \u2248 o3, taking photos to solve math problems", "support": "image", "tip": "Qwen o1"}, {"id": "DeepSeek-R1", "object": "model", "model": "DeepSeek-R1", "created": 1744090984000, "owned_by": "DeepSeek", "name": "DeepSeek R1", "description": "Deep thinking,mathematical and writing abilities \u2248 o3", "support": "text", "tip": "DeepSeek R1"}, {"id": "Llama3.3-70B", "object": "model", "model": "Llama3.3-70B", "created": 1744090984000, "owned_by": "Llama3.3", "name": "Llama3.3", "description": "Suitable for most tasks", "support": "text", "tip": "Llama3.3"}], "version": "0.1.125", "provider": "DeGPT", "name": "DeGPT", "default_locale": "en-US", "status": true, "time": 0}
 #    """
 #    ress = chat_completion_message(user_prompt=msg)
 #    print(ress)
 #    print(type(ress))
 #    print("\r\n----------\r\n\r\n")
