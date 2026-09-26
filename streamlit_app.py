import os
import sys

# Ensure root directory is in python path and execute main application
sys.path.append(os.path.dirname(__file__))

# Run app.py main logic
with open(os.path.join(os.path.dirname(__file__), "app.py"), "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, "app.py", "exec"))
