"""
Run the Sequence Baseball API server
Usage: py -3 run_api.py
"""
import uvicorn

if __name__ == "__main__":
    print("Starting Sequence Baseball API...")
    print("API docs available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )








