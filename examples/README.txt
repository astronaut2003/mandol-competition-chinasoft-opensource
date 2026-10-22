Public examples use freshly authored synthetic input, never production accounts, conversations, memory databases or indexes.

Version/provenance example (uses only freshly created synthetic records):
  .venv/bin/python examples/memory_lifecycle.py --username YOUR_USER --password-file /secure/password.txt
Optional --wait-higher 60 waits for actual configured models. Pending/failed jobs
remain visible; this example does not substitute generated memory with mock data.
