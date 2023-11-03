import httpx

url = 'http://127.0.0.1:8000/api/v1/chat/chat'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

sessionID = "0192839202"
api_key = "0x192839202"

while True: 
    textIn = input("\n\nEnter: ")
    payload = {
        'query': f"{textIn}",
        'session_id': f'{sessionID}',
        'api_key': f'{api_key}',
        'model': 'string',
        'temperature': 0,
        'max_tokens': 0
    }

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
        if response.status_code == 200:
            for chunk in response.iter_text():
                print(chunk, end='', flush=True)   
        else:
            print(f"Failed: {response.status_code}")
            
    # with httpx.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
    #     if response.status_code == 200:
    #         for chunk in response.iter_text():
    #             print(chunk, end='')  
    #     else:
    #         print(f"Failed: {response.status_code}")
            