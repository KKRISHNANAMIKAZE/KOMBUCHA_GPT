from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting simple server on {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)