import time
import json
from queue import Queue
from threading import Thread
import torch
import sys
from benchmark_image2text import monitor_gpu_memory, copy_images_to_directory
import os


    
from server import extract_text, syntax_check, grammar_check, set_device

def run_server_tasks_with_monitoring(memory_queue, images_dir, batch_size, device):
    """Run the server tasks and monitor GPU memory usage."""
    gpu_memory_log = []
    # Start GPU monitoring thread
    monitor_thread = Thread(target=monitor_gpu_memory, args=(memory_queue,))
    monitor_thread.daemon = True
    monitor_thread.start()
    # Timing and execution
    start_time = time.time()
    # Step 1: Extract Text
    extracted_text = extract_text(images_dir, batch_size, device)
    # Step 2: Syntax Check
    big_text = '\n'.join(f"{item}" for item in extracted_text)
    syntax_fixed_text = syntax_check(big_text)
    # Step 3: Grammar Check
    grammar_fixed_text = grammar_check(syntax_fixed_text, device)
    end_time = time.time()
    # Gather GPU memory logs
    while not memory_queue.empty():
        gpu_memory_log.append(memory_queue.get())
    # Results
    total_execution_time = end_time - start_time
    result = {
        "execution_time": total_execution_time,
        "gpu_memory_log": gpu_memory_log,
    }
    return result
def benchmark_server(test_file, source_image_dir, temp_image_dir, output_file, batch_sizes, deviceName):
    """Benchmark the full server execution process."""
    # copy_images_to_directory(test_file, source_image_dir, temp_image_dir)
    all_results = {}
    for batch_size in batch_sizes:
        print(f"------------------------------------------------"
              f"Running benchmark for batch_size={batch_size}...")
        memory_queue = Queue()
        results = run_server_tasks_with_monitoring(memory_queue, temp_image_dir, batch_size, torch.device(deviceName))
        all_results[f"{batch_size} batches"] = results
        # Save results
    with open(output_file, 'w') as file:
        json.dump(all_results, file, indent=4)
    print(f"Benchmark completed. Results saved to {output_file}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark text extraction performance.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test file containing image names and ground truth text.")
    parser.add_argument("--source_image_dir", type=str, required=True, help="Directory containing the source images.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file where results will be saved.")
    parser.add_argument("--temp_image_dir", type=str, required=True,
                        help="Temporary directory where images will be copied for processing.")
    parser.add_argument("--batch_sizes", type=int, nargs='+', required=True,
                        help="List of batch sizes to test.")
    parser.add_argument("--device",  type=str, required=False, default="cuda",
                        help="The device to be used for benchmark.")
    args = parser.parse_args()
    # Run the benchmark
    benchmark_server(args.test_file, args.source_image_dir, args.temp_image_dir, args.output_file, args.batch_sizes, args.device)