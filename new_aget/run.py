"""
Simple CLI runner for the LangGraph agent pipeline.
Usage:
  python run.py input.json
Or echo JSON to stdin:
  cat input.json | python run.py
"""

import sys
import json
from graph import build_and_run_graph

def load_input(path_or_dash):
    if path_or_dash == "-":
        return json.load(sys.stdin)
    with open(path_or_dash, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python run.py <input.json>  (use - to read stdin)")
    #     sys.exit(2)
    # inp_path = sys.argv[1]
    # payload = load_input(inp_path)
    payload =  { 
"url": "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ==", 
"questions": ["""Go to the website and start the challenge. Complete the challenge and return 
the answers for the following questions: What is the challenge name?""" 
] 
}
    result = build_and_run_graph(payload)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
