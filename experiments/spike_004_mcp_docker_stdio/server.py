"""Disposable line-delimited JSON-RPC echo; stdout is protocol only."""
import json
import sys

for line in sys.stdin:
    try:
        request=json.loads(line); response={"jsonrpc":"2.0","id":request.get("id"),"result":{"echo":request.get("method")}}
        print(json.dumps(response), flush=True)
    except json.JSONDecodeError as error:
        print(f"invalid protocol input: {error}", file=sys.stderr, flush=True)
