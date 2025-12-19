import requests
import json

def test_table_rag_request():
    """
    Test the TableRAG request processing.
    """
    table_name_list = ["List_of_National_Football_League_rushing_yards_leaders_0"]
    query = "What is the middle name of the player with the second most National Football League career rushing yards ?"
    url = "http://localhost:5000/get_tablerag_response"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "query": query,
        "table_name_list": table_name_list
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        print("Response Data:", json.dumps(response_data, ensure_ascii=False))
    else:
        print(f"Request failed with status code {response.status_code}. Response: {response.text}")

if __name__ == "__main__":  
    test_table_rag_request()