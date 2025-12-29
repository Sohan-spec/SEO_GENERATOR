from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Simple Test API")

# -----------------------
# Data model
# -----------------------
class Item(BaseModel):
    name: str
    price: float
    quantity: int

# -----------------------
# Root endpoint (GET)
# -----------------------
@app.get("/")
def home():
    return {"message": "API is working 🚀"}

# -----------------------
# Test POST endpoint
# -----------------------
@app.post("/add-item")
def add_item(item: Item):
    total_cost = item.price * item.quantity

    return {
        "status": "success",
        "item": item,
        "total_cost": total_cost
    }

# -----------------------
# Simple GET with params
# -----------------------
@app.get("/hello/{username}")
def hello_user(username: str):
    return {"message": f"Hello {username}!"}
