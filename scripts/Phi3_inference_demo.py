# phi3_demo_terminal.py
import requests
import time
import GPUtil

SERVER_URL = "http://127.0.0.1:8000/ask"  # running RAG server

def get_gpu_status():
    """Return current GPU memory and utilization using GPUtil."""
    try:
        gpu = GPUtil.getGPUs()[0]
        return f"{gpu.name} | Mem: {gpu.memoryUsed}/{gpu.memoryTotal} MB | Load: {gpu.load*100:.1f}%"
    except Exception:
        return "GPU not detected"

def main():
    print("=== Phi-3 RAG QA Live Demo ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting demo.")
            break

        print(f"[GPU Status] Before request: {get_gpu_status()}")
        start_time = time.time()
        try:
            response = requests.post(SERVER_URL, json={"question": question})
            elapsed = time.time() - start_time

            if response.status_code == 200:
                answer = response.json().get("answer", "No answer returned.")
                print(f"Answer: {answer}")
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
            continue

        print(f"[GPU Status] After request: {get_gpu_status()}")
        print(f"[Time taken] {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    main()
