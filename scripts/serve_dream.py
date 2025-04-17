import torch
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
import time
from collections import deque

# Helper function to run model inference
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="Dream Model API", description="Simple API for Dream model generation")

# Initialize model
try:
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda").eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

# Request queue and processing lock
request_queue = deque()
model_lock = asyncio.Lock()

class GenerationRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    steps: int = 512
    alg: str = "entropy"
    alg_temp: float = 0.0
    timeout: int = 300  # Default timeout in seconds

class GenerationResponse(BaseModel):
    result: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    try:
        # Try to acquire the lock with timeout
        async with asyncio.timeout(request.timeout):
            async with model_lock:  # This will wait if another request is processing
                # Run the model inference
                result = await run_model_inference(
                    request.messages,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    steps=request.steps,
                    alg=request.alg,
                    alg_temp=request.alg_temp
                )
                return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,  # Request Timeout
            detail=f"Request timed out after {request.timeout} seconds. The server is busy processing other requests."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


async def run_model_inference(
    messages, max_new_tokens=512, temperature=0.2, top_p=0.95, steps=512, alg="entropy", alg_temp=0.0
):
    # Run the compute-intensive part in a thread pool
    def run_inference():
        # Process input
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")
        
        # Generate text
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
        )
        
        # Process output
        generations = [
            tokenizer.decode(g[len(p):].tolist())
            for p, g in zip(input_ids, output.sequences)
        ]
        
        return generations[0].split(tokenizer.eos_token)[0]
    
    # This runs the function in a threadpool, keeping the async event loop responsive
    return await run_in_threadpool(run_inference)

# Custom middleware to handle server load
@app.middleware("http")
async def add_load_info_header(request: Request, call_next):
    # Calculate number of active model operations
    is_locked = model_lock.locked()
    
    response = await call_next(request)
    
    # Add custom headers to indicate server load
    response.headers["X-Server-Busy"] = str(is_locked)
    
    return response

# Endpoint to check server status
@app.get("/status")
async def check_status():
    return {
        "status": "busy" if model_lock.locked() else "available",
        "model_loaded": model is not None and tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)