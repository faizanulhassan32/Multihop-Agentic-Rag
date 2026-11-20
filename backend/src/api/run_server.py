import os
import sys
import uvicorn

# Add project root and libs to PYTHONPATH so backend modules can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
libs_path = os.path.abspath(os.path.join(project_root, "backend", "libs"))
sys.path.append(project_root)
sys.path.append(libs_path)

# Import the FastAPI app defined in the API module
from backend.src.api.main import app

if __name__ == "__main__":
    # Run the FastAPI server on port 2024 (as expected by the frontend)
    uvicorn.run(app, host="0.0.0.0", port=2024, reload=True)
