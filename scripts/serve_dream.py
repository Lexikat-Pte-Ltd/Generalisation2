import torch
import asyncio
import logging
import gc
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
import time
from fastapi.concurrency import run_in_threadpool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dream Model API", description="Simple API for Dream model generation"
)

# Global variables
model = None
tokenizer = None
model_path = "Dream-org/Dream-v0-Instruct-7B"
model_lock = asyncio.Lock()
LOCK_TIMEOUT = 60  # seconds
MAX_CONCURRENT_REQUESTS = 3  # adjust based on your GPU capacity
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Health monitoring variables
last_successful_inference = None
inference_count = 0
failed_inference_count = 0
MAX_FAILURES_BEFORE_RELOAD = 5


class Message(BaseModel):
    role: str
    content: str


class GenerationRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    steps: int = Field(default=10, ge=1, le=50)
    alg: str = Field(default="entropy", pattern="^(entropy|uniform)$")
    alg_temp: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout: int = Field(default=300, ge=10, le=600)  # Default timeout in seconds


class GenerationResponse(BaseModel):
    result: str
    processing_time: float
    queue_time: float
    progression: List[str]


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    cuda_available: bool
    gpu_utilization: Optional[float]
    gpu_memory_allocated: Optional[float]
    gpu_memory_total: Optional[float]
    active_requests: int
    inference_count: int
    last_successful_inference: Optional[float]
    failed_inference_count: int


async def load_model():
    """Initialize model and tokenizer."""
    global model, tokenizer

    try:
        logger.info("Loading tokenizer from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        logger.info("Loading model from %s", model_path)
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

        if torch.cuda.is_available():
            logger.info("Moving model to CUDA")
            model = model.to("cuda").eval()
        else:
            logger.warning("CUDA not available, using CPU")
            model = model.eval()

        logger.info("Model and tokenizer loaded successfully")
        return True
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        model = None
        tokenizer = None
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    await load_model()


def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    if not torch.cuda.is_available():
        return None, None, None

    try:
        # Get memory allocation stats
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        # For utilization, we'd typically use pynvml or similar
        # This is a simple approximation
        utilization = allocated / total * 100 if total > 0 else 0

        return utilization, allocated, total
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {str(e)}")
        return None, None, None


async def run_model_inference(
    messages,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.95,
    steps=10,
    alg="entropy",
    alg_temp=0.0,
):
    """Run model inference in a threadpool."""
    global inference_count, failed_inference_count, last_successful_inference

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Run the compute-intensive part in a thread pool
    def run_inference():
        global inference_count, failed_inference_count, last_successful_inference

        try:
            # Process input
            inputs = tokenizer.apply_chat_template(  # type: ignore
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = inputs.input_ids.to(device=device)  # type: ignore
            attention_mask = inputs.attention_mask.to(device=device)  # type: ignore

            progression = []

            def generation_tokens_hook_func(step, x, logits):
                progression.append(
                    f"Step {step}: "
                    + tokenizer.decode(x[0].tolist())  # type: ignore
                    .split(tokenizer.eos_token)[0]  # type: ignore
                    .replace(tokenizer.mask_token, "<MaskToken>"),  # type: ignore
                )
                return x

            # Generate text
            output = model.diffusion_generate(  # type: ignore
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
                generation_tokens_hook_func=generation_tokens_hook_func,
            )

            # Process output
            generations = [
                tokenizer.decode(g[len(p) :].tolist())  # type: ignore
                for p, g in zip(input_ids, output.sequences)
            ]

            result = generations[0].split(tokenizer.eos_token)[0]  # type: ignore

            # Update metrics
            inference_count += 1
            last_successful_inference = time.time()

            # Force garbage collection to free memory
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            return result, progression

        except Exception as e:
            failed_inference_count += 1
            logger.error(f"Inference failed: {str(e)}")

            # Check if we need to reload the model
            if failed_inference_count >= MAX_FAILURES_BEFORE_RELOAD:
                logger.warning(
                    f"Too many failures ({failed_inference_count}), scheduling model reload"
                )
                # We'll handle model reloading in the main request handler

            raise

    # This runs the function in a threadpool, keeping the async event loop responsive
    return await run_in_threadpool(run_inference)


async def reload_model_if_needed(background_tasks: BackgroundTasks):
    """Check if model needs reloading and schedule it if necessary."""
    global failed_inference_count

    if failed_inference_count >= MAX_FAILURES_BEFORE_RELOAD:
        logger.warning("Scheduling model reload due to excessive failures")

        async def reload_task():
            global failed_inference_count

            async with model_lock:
                logger.info("Reloading model due to excessive failures")
                # Clear CUDA cache and force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Reset failure count and reload model
                failed_inference_count = 0
                await load_model()
                logger.info("Model reloaded successfully")

        background_tasks.add_task(reload_task)


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text from the model."""
    if model is None or tokenizer is None:
        await load_model()
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model failed to load")

    # Check if we need to reload the model due to past failures
    await reload_model_if_needed(background_tasks)

    queue_start_time = time.time()

    try:
        # Try to acquire semaphore with timeout
        semaphore_acquired = False
        try:
            async with request_semaphore:
                semaphore_acquired = True

                # Now try to acquire the model lock with timeout
                try:

                    async def inner_fn():
                        async with model_lock:
                            queue_time = time.time() - queue_start_time
                            processing_start_time = time.time()

                            # Run the model inference
                            result, progression = await run_model_inference(
                                request.messages,
                                max_new_tokens=request.max_new_tokens,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                steps=request.steps,
                                alg=request.alg,
                                alg_temp=request.alg_temp,
                            )

                            processing_time = time.time() - processing_start_time

                            return {
                                "result": result,
                                "processing_time": processing_time,
                                "queue_time": queue_time,
                                "progression": progression,
                            }

                    return await asyncio.wait_for(inner_fn(), timeout=request.timeout)
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model lock acquisition timed out after {LOCK_TIMEOUT} seconds",
                    )
        except asyncio.TimeoutError:
            if not semaphore_acquired:
                raise HTTPException(
                    status_code=503,
                    detail="Server is at maximum capacity. Try again later.",
                )

        # If we reach here with semaphore_acquired=True, it means we got past the semaphore
        # but had some other error, which would have been raised already

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def check_status():
    """Check server status with detailed metrics."""
    utilization, allocated, total = get_gpu_stats()

    return {
        "status": "busy" if model_lock.locked() else "available",
        "model_loaded": model is not None and tokenizer is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_utilization": utilization,
        "gpu_memory_allocated": allocated,
        "gpu_memory_total": total,
        "active_requests": MAX_CONCURRENT_REQUESTS - request_semaphore._value,
        "inference_count": inference_count,
        "last_successful_inference": last_successful_inference,
        "failed_inference_count": failed_inference_count,
    }


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"},
        )

    # Check if there have been too many failures
    if failed_inference_count >= MAX_FAILURES_BEFORE_RELOAD:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Too many inference failures"},
        )

    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global model, tokenizer

    logger.info("Shutting down, cleaning up resources")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear model and tokenizer
    model = None
    tokenizer = None
    gc.collect()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6969)
