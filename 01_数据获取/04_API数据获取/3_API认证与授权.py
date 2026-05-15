# 数据来源: 模拟API认证场景（无真实密钥泄露风险）

import json
import hashlib
import hmac
import time
from pathlib import Path

MOCK_USER_DATA = {
    "id": 1,
    "name": "张三",
    "email": "zhangsan@example.com",
    "role": "admin",
}

MOCK_PROTECTED_DATA = [
    {"id": 1, "project": "数据分析项目A", "status": "进行中"},
    {"id": 2, "project": "数据分析项目B", "status": "已完成"},
]


def demo_api_key():
    print("=" * 50)
    print("1. API Key认证")
    print("=" * 50)

    api_key = "demo_api_key_12345"

    headers_with_key = {"X-API-Key": api_key}
    print("方式一 - 请求头传递:")
    print(f"  Headers: {headers_with_key}")

    url_with_key = f"https://api.example.com/data?api_key={api_key}"
    print("\n方式二 - Query参数传递:")
    print(f"  URL: {url_with_key}")

    print("\n模拟验证:")
    if headers_with_key["X-API-Key"] == "demo_api_key_12345":
        print(f"  ✓ API Key验证通过，返回数据: {MOCK_USER_DATA}")
    else:
        print("  ✗ API Key无效")


def demo_bearer_token():
    print("\n" + "=" * 50)
    print("2. Bearer Token认证")
    print("=" * 50)

    token_payload = {"user_id": 1, "role": "admin", "exp": int(time.time()) + 3600}
    fake_token = (
        "eyJhbGciOiJIUzI1NiJ9."
        + hashlib.sha256(json.dumps(token_payload).encode()).hexdigest()[:32]
        + ".fake_signature"
    )

    headers = {"Authorization": f"Bearer {fake_token}"}
    print(f"Token: {fake_token[:50]}...")
    print(f"请求头: Authorization: Bearer {fake_token[:30]}...")

    print("\n模拟请求流程:")
    print("  1. 客户端携带Token发起请求")
    print("  2. 服务端验证Token有效性")
    print("  3. 验证通过返回受保护数据")
    print(f"  ✓ 返回数据: {MOCK_PROTECTED_DATA}")


def demo_basic_auth():
    print("\n" + "=" * 50)
    print("3. HTTP Basic认证")
    print("=" * 50)

    import base64

    username = "admin"
    password = "secret123"

    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode()).decode()

    headers = {"Authorization": f"Basic {encoded}"}
    print(f"用户名: {username}")
    print(f"密码: {'*' * len(password)}")
    print(f"Base64编码: {encoded}")
    print(f"请求头: Authorization: Basic {encoded}")

    decoded = base64.b64decode(encoded).decode()
    print(f"\n解码验证: {decoded}")
    print("⚠ Basic认证仅适用于HTTPS，否则密码可被截获")


def demo_oauth2_concept():
    print("\n" + "=" * 50)
    print("4. OAuth2基础概念")
    print("=" * 50)

    print("OAuth2授权码流程:")
    print("  1. 客户端重定向用户到授权服务器")
    auth_url = (
        "https://auth.example.com/authorize?"
        "response_type=code&client_id=myapp&"
        "redirect_uri=https://myapp.com/callback&"
        "scope=read+write"
    )
    print(f"     授权URL: {auth_url[:70]}...")

    print("\n  2. 用户授权后，回调携带授权码")
    callback_url = "https://myapp.com/callback?code=AUTH_CODE_123"
    print(f"     回调URL: {callback_url}")

    print("\n  3. 客户端用授权码换取Access Token")
    token_request = {
        "grant_type": "authorization_code",
        "code": "AUTH_CODE_123",
        "client_id": "myapp",
        "client_secret": "app_secret",
        "redirect_uri": "https://myapp.com/callback",
    }
    print(f"     Token请求体: {json.dumps(token_request, indent=4)}")

    mock_token_response = {
        "access_token": "mock_access_token_xyz",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "mock_refresh_token_abc",
    }
    print(f"\n  4. 获取到Token:")
    print(f"     {json.dumps(mock_token_response, indent=4)}")


def demo_hmac_signature():
    print("\n" + "=" * 50)
    print("5. HMAC签名认证")
    print("=" * 50)

    secret_key = "my_secret_key_123"
    timestamp = str(int(time.time()))
    method = "GET"
    path = "/api/v1/data"

    string_to_sign = f"{method}\n{path}\n{timestamp}"
    print(f"签名字符串: {string_to_sign}")

    signature = hmac.new(
        secret_key.encode(), string_to_sign.encode(), hashlib.sha256
    ).hexdigest()

    headers = {
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "X-Access-Key": "my_access_key",
    }
    print(f"签名结果: {signature}")
    print(f"请求头: {json.dumps(headers, indent=2)}")

    print("\n验证签名:")
    expected = hmac.new(
        secret_key.encode(), string_to_sign.encode(), hashlib.sha256
    ).hexdigest()
    if hmac.compare_digest(signature, expected):
        print("  ✓ 签名验证通过")
    else:
        print("  ✗ 签名验证失败")


def demo_session_auth():
    print("\n" + "=" * 50)
    print("6. requests.Session持久认证")
    print("=" * 50)

    import requests

    session = requests.Session()
    session.headers.update({"Authorization": "Bearer mock_token"})
    session.headers.update({"X-API-Key": "demo_key"})

    print("Session自动携带的Headers:")
    for key, val in session.headers.items():
        if key in ("Authorization", "X-API-Key"):
            print(f"  {key}: {val[:30]}..." if len(val) > 30 else f"  {key}: {val}")

    print("\n优势: 所有请求自动携带认证信息，无需每次手动设置")


if __name__ == "__main__":
    demo_api_key()
    demo_bearer_token()
    demo_basic_auth()
    demo_oauth2_concept()
    demo_hmac_signature()
    demo_session_auth()
