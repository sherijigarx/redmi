import requests
import os

class CorcelAPI:
    def __init__(self):
        self.base_url = "https://api.corcel.io/v1/text/cortext/chat"
        self.api_key = os.getenv('API_KEY')
        if self.api_key is None:
            raise Exception("Please set the API_KEY environment variable")
        self.headers = {
            "Authorization": self.api_key,
            "accept": "application/json",
            "content-type": "application/json"
        }
    
    def post_request(self, data):
        response = requests.post(self.base_url, headers=self.headers, json=data)
        # Check the HTTP status code
        if response.status_code == 200:
            # If status code is 200, parse the response
            json_data = response.json()
            content = json_data[0]['choices'][0]['delta']['content']
            return content
        else:
            # Handle other failure codes
            return None
    
    def get_TTS(self):
        data = {
            "messages": [{"role": "user", "content": "random meaningful text phrase in less than 32 words"}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)
    
    def get_VC(self):
        data = {
            "messages": [{"role": "user", "content": "random meaningful text phrase in less than 32 words"}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)
    
    def get_TTM(self):
        data = {
            "messages": [{"role": "user", "content": "random Music generation phrase for AI music generation model in less than 32 words"}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)