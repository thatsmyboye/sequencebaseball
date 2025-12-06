"""
Start the Sequence Baseball API
Automatically selects lite or full mode based on environment
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    api_mode = os.environ.get("API_MODE", "full").lower()
    
    # Select API module based on mode
    if api_mode == "lite":
        app_module = "api.main_lite:app"
        print("Starting in LITE mode (lower memory usage)")
    else:
        app_module = "api.main:app"
        print("Starting in FULL mode")
    
    print(f"Port: {port}")
    print(f"API docs: http://localhost:{port}/docs")
    print("-" * 40)
    
    uvicorn.run(
        app_module,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
