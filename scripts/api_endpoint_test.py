# scripts/api_endpoint_test.py
import requests
import json
import sys

def main():
    url = "http://127.0.0.1:8000/api/v1/chat"
    
    payload = {
        "message": "What is the difference between overfitting and underfitting?",
        "history": []
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        print("\n=== API RESPONSE ===")
        print(f"Answer: {data.get('answer')}")
        
        sources = data.get("sources", [])
        print(f"\nSources ({len(sources)}):")
        for i, s in enumerate(sources, 1):
            print(f"  {i}. {s[:100]}...")
            
        judge = data.get("judge_result")
        if judge:
            print(f"\nJudge Verdict: {judge.get('verdict')}")
            print(f"Scores: {judge.get('scores')}")
        else:
            print("\nJudge: None (Check if judge is enabled/loaded)")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure 'scripts/run_backend.py' is running!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
