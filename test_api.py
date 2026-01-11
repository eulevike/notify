"""
Test script to verify z.ai API multimodal format
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZAI_API_KEY")

# Different possible endpoints
ENDPOINTS = [
    "https://api.z.ai/api/paas/v4/chat/completions",
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
]

def test_multimodal_format_1(endpoint, model):
    """Test OpenAI-style format with image_url."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
        ]
    }]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 50}
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"✓ Format 1 (OpenAI-style): WORKS")
            return True
        else:
            print(f"✗ Format 1 (OpenAI-style): {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ Format 1 (OpenAI-style): {str(e)}")
        return False

def test_multimodal_format_2(endpoint, model):
    """Test Zhipu AI format with separate image parameter."""
    messages = [{
        "role": "user",
        "content": "What do you see?"
    }]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 50,
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"✓ Format 2 (separate image param): WORKS")
            return True
        else:
            print(f"✗ Format 2 (separate image param): {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ Format 2 (separate image param): {str(e)}")
        return False

def test_multimodal_format_3(endpoint, model):
    """Test content array with image type."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image", "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}
        ]
    }]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 50}
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"✓ Format 3 (type:image): WORKS")
            return True
        else:
            print(f"✗ Format 3 (type:image): {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ Format 3 (type:image): {str(e)}")
        return False

def test_multimodal_format_4(endpoint, model):
    """Test image_url directly in content."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}
        ]
    }]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 50}
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"✓ Format 4 (image_url direct): WORKS")
            return True
        else:
            print(f"✗ Format 4 (image_url direct): {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ Format 4 (image_url direct): {str(e)}")
        return False

print("="*60)
print("Testing z.ai API Multimodal Formats")
print("="*60)

# Test glm-4-plus first (we know this works)
print("\n=== Testing glm-4-plus (known working) ===")
for endpoint in ENDPOINTS:
    print(f"\nTesting endpoint: {endpoint.split('//')[1].split('/')[0]}")
    print("-"*60)
    test_multimodal_format_2(endpoint, "glm-4-plus")

# Test if GLM-4.6V exists
print("\n\n=== Testing GLM-4V variants ===")
for endpoint in ENDPOINTS:
    print(f"\nTesting endpoint: {endpoint.split('//')[1].split('/')[0]}")
    print("-"*60)
    test_multimodal_format_2(endpoint, "glm-4v")
    test_multimodal_format_2(endpoint, "glm-4.6v")
    test_multimodal_format_2(endpoint, "glm-4.7")

print("\n" + "="*60)
print("Test complete. Use the working format in monitor.py")
print("="*60)
