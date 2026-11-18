"""Public version and hashing helpers only; no solutions, labels, or evaluators."""
import hashlib
import json

VERSION='cpu-pilot-1.3'
RELEASE_DIR='eight_domains_v4'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
