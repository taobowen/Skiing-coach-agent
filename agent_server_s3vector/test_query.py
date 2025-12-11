# test_query.py
from agent_server import query_engine

def main():
    resp = query_engine.query("How can I stay out of the back seat on skis?")
    print("=== RAW RESPONSE ===")
    print(resp)
    print("=== SOURCE NODES ===")
    for n in resp.source_nodes:
        print("----")
        print(n.text[:300])

if __name__ == "__main__":
    main()
