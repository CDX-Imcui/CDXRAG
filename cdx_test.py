import requests
from http import HTTPStatus

url = "http://127.0.0.1:5000/get_tablerag_response"
response = requests.post(url,
                         json={"query": "统计 a1_team_mexico_0 这张表有多少行", "table_name_list": ["a1_team_mexico_0"]}
                         , headers={"Content-Type": "application/json"}
                         )

try:
    if response.status_code == HTTPStatus.OK:
        print("✅ API Key 有效")
    elif response.status_code == 402:
        print("❌ 余额不足或欠费")
    elif response.status_code == 403:
        print("❌ API Key 无效或权限不足")
    else:
        print(f"⚠️ 其他错误: {response.status_code} - {response.text}")
    print("resp.text:", response.text)
    print(response.json())
except Exception as e:
    print(f"  🚨 请求异常: {e}")

