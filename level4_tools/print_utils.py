from rich.console import Console
console = Console()
print = console.print

import time
import asyncio
import functools
import inspect

def time_it(func):
    """A universal decorator to measure execution time for both sync and async functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the function is a coroutine function (async def)
        if inspect.iscoroutinefunction(func):
            # Define and return an async wrapper to handle the coroutine
            async def async_wrapper():
                start_time = time.perf_counter()
                result = await func(*args, **kwargs) # Await the coroutine
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Async function '{func.__name__}' took {elapsed_time:.4f} seconds.")
                return result
            return async_wrapper()
        else:
            # Use the original synchronous logic
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Sync function '{func.__name__}' took {elapsed_time:.4f} seconds.")
            return result
    return wrapper
