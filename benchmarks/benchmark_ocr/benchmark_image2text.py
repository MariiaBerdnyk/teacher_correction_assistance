import json
import os
import shutil
import time
import argparse
from queue import Queue

import torch
from rouge import Rouge
from subprocess import Popen, PIPE
from threading import Thread

import subprocess as sp


def get_free_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_info

# Function to copy images to a separate directory
def copy_images_to_directory(test_file, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    i = 0

    with open(test_file, 'r') as file:
        for line in file:
            if i > 1000:
                break
            i += 1
            image_file, _ = line.strip().split('\t')
            image_path = os.path.join(source_dir, image_file)
            if os.path.exists(image_path):
                shutil.copy(image_path, target_dir)
            else:
                print(f"Image {image_path} does not exist, skipping.")
    print(f"All images copied to {target_dir}")


def read_stream(stream, output_list):
    """Read a stream line-by-line and store it in the output list."""
    for line in iter(stream.readline, ''):
        output_list.append(line.strip())
    stream.close()


def monitor_process_with_output(process, memory_queue, gpu_memory_log):
    """Monitor the process while periodically checking GPU memory and reading process output."""
    stdout_lines = []
    stderr_lines = []

    # Start threads to read stdout and stderr
    stdout_thread = Thread(target=read_stream, args=(process.stdout, stdout_lines))
    stderr_thread = Thread(target=read_stream, args=(process.stderr, stderr_lines))
    stdout_thread.start()
    stderr_thread.start()

    while True:
        # Check for process termination
        retcode = process.poll()
        if retcode is not None:  # Process has terminated
            break

        # Periodically check GPU memory
        if not memory_queue.empty():
            gpu_memory_log.append(memory_queue.get())

        time.sleep(0.1)  # Prevent tight looping

    # Ensure threads finish reading
    stdout_thread.join()
    stderr_thread.join()

    return stdout_lines, stderr_lines



# Function to call the image2text_recognition.py script and get generated texts
def run_image2text_recognition_with_monitoring(image_dir, model_name, batch_size, memory_queue, load_local_model, device):
    command = ["python", "image2text_recognition.py",
               "--load_local_model", str(load_local_model),
               "--image_dir", image_dir,
               "--device", device,
               "--model_name", model_name,
               "--batch_size", str(batch_size)]
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)

    gpu_memory_log = []
    stdout_lines, stderr_lines = monitor_process_with_output(process, memory_queue, gpu_memory_log)

    if process.returncode != 0:
        raise RuntimeError(f"Error: {' '.join(stderr_lines)}")

    return stdout_lines, gpu_memory_log


# Function to calculate ROUGE scores
def calculate_rouge_scores(predicted, reference):
    rouge = Rouge()
    scores = rouge.get_scores(predicted, reference)
    return scores[0]  # return the first (and only) score dictionary


# Threaded GPU memory monitoring
def monitor_gpu_memory(queue, interval=1):
    while True:
        memory_info = get_free_gpu_memory()
        queue.put(memory_info)
        time.sleep(interval)


# Function to benchmark the process
def benchmark(test_file, source_image_dir, model_name, output_file, temp_image_dir, batch_sizes, load_local_model, device):
    results_summary = []
    # Copy images to a new directory
    copy_images_to_directory(test_file, source_image_dir, temp_image_dir)

    for batch_size in batch_sizes:
        print(f"Running benchmark for batch_size={batch_size}...")
        memory_queue = Queue()

        # Start GPU monitoring thread
        if device != "cpu":
            torch.cuda.empty_cache()
            monitor_thread = Thread(target=monitor_gpu_memory, args=(memory_queue,))
            monitor_thread.daemon = True
            monitor_thread.start()

        # Start timing the execution
        start_time = time.time()

        # Monitor GPU memory usage during execution
        generated_texts, gpu_memory_log = run_image2text_recognition_with_monitoring(temp_image_dir, model_name, batch_size, memory_queue, load_local_model, device)

        end_time = time.time()

        raw_generated_texts = "".join(generated_texts[:-1]).strip()

        generated_texts = json.loads(raw_generated_texts)

        # Calculate ROUGE scores
        rouge_scores = []
        results = []
        with open(test_file, 'r') as file:
            for idx, line in enumerate(file):
                if idx > 1000:
                    break
                image_file, expected_text = line.strip().split('\t')
                generated_text = generated_texts[idx] if idx < len(generated_texts) else ""

                rouge_score = calculate_rouge_scores(generated_text, expected_text)
                rouge_scores.append(rouge_score)
                results.append({
                    'image_name': image_file,
                    'generated_text': generated_text,
                    'expected_text': expected_text,
                    'rouge_score': rouge_score,
                })

        # Save results for this batch size
        total_execution_time = end_time - start_time
        avg_rouge_score = {
            'rouge-1': sum(r['rouge-1']['f'] for r in rouge_scores) / len(rouge_scores),
            'rouge-2': sum(r['rouge-2']['f'] for r in rouge_scores) / len(rouge_scores),
            'rouge-l': sum(r['rouge-l']['f'] for r in rouge_scores) / len(rouge_scores),
        }
        results_summary.append({
            'batch_size': batch_size,
            'execution_time': total_execution_time,
            'avg_rouge_score': avg_rouge_score,
            'gpu_memory_log': gpu_memory_log,
        })

        # Save detailed results
        with open(output_file, 'a') as output:
            output.write(f"Batch Size: {batch_size}\n")
            output.write(f"Total Execution Time: {total_execution_time:.2f} seconds\n")
            output.write(f"Average ROUGE Scores: {avg_rouge_score}\n")
            output.write("GPU Memory Log (MiB):\n")
            for memory in gpu_memory_log:
                output.write(f"{memory}\n")
            output.write("-" * 50 + "\n")

    print(f"Benchmark completed. Results saved to {output_file}")
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark text extraction performance.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test file containing image names and ground truth text.")
    parser.add_argument("--source_image_dir", type=str, required=True, help="Directory containing the source images.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file where results will be saved.")
    parser.add_argument("--temp_image_dir", type=str, required=True,
                        help="Temporary directory where images will be copied for processing.")
    parser.add_argument("--batch_sizes", type=int, nargs='+', required=True,
                        help="List of batch sizes to test.")
    parser.add_argument("--device",  type=str, required=False, default="cuda",
                        help="The device to be used for benchmark.")
    parser.add_argument("--load_local_model",  type=bool, required=False, default=False,
                        help="Load the base model or one from hf repository.")
    args = parser.parse_args()

    # Run the benchmark with different batch sizes
    benchmark(args.test_file, args.source_image_dir, args.model_name, args.output_file, args.temp_image_dir, args.batch_sizes, args.load_local_model, args.device)
