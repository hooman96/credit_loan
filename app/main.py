from fastapi import FastAPI

app = FastAPI(title="Hello World Service")


@app.get("/")
def read_root():
    return {"message": "Hello World"}
