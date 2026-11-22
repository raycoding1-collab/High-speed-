"""Smallest possible Modal GPU test: start one L4 container, report the GPU, exit (a few seconds of GPU time)."""
import modal

app = modal.App("cr450-gpu-check")


@app.function(gpu="L4", timeout=300)
def gpu_info() -> str:
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main():
    print(f"[modal-check] GPU OK: {gpu_info.remote()}")
