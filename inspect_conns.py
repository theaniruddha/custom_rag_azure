# inspect_connections.py
import os, json
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()
endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
if not endpoint:
    raise SystemExit("FOUNDRY_PROJECT_ENDPOINT missing in .env")

credential = DefaultAzureCredential()
client = AIProjectClient(endpoint=endpoint, credential=credential)

print("Listing project connections...")
conns = list(client.connections.list())

if not conns:
    print("No connections found in project.")
else:
    for c in conns:
        # Print compact summary and full JSON for inspection
        print("----")
        print(f"id: {c.get('id')}")
        print(f"name: {c.get('name')}")
        print(f"connection_type: {c.get('connection_type')}")
        print(f"is_default: {c.get('is_default', False)}")
        print(json.dumps(c, indent=2, default=str))

# Try to pick a Cognitive Search connection automatically
candidates = []
for c in conns:
    ct = str(c.get("connection_type", "")).lower()
    if any(k in ct for k in ("search", "cognitiveservices", "cognitive", "azure_ai_search", "azureai_search", "cognitivesearch")):
        candidates.append(c)

if candidates:
    chosen = candidates[0]
    print("\nSelected connection id:", chosen.get("id"))
else:
    print("\nNo Cognitive Search connection candidate found. Create one in the Foundry UI or use a connection id manually.")
