import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import app, init_app

if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
