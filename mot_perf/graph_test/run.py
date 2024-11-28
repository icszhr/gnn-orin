import subprocess
import time

# 定义执行每个脚本的函数
def run_script(script_name):
    print(f"Running {script_name}...")
    start_time = time.time()
    
    # 执行脚本并等待结束
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    
    # 获取执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 打印执行时间
    print(f"{script_name} executed in {execution_time:.4f} seconds")
    
    # 返回执行时间和脚本的输出
    return execution_time, result.stdout

# 定义一个函数来运行所有脚本并计算平均执行时间
def run_all_scripts():
    scripts = ["infer.py", "pin_infer.py", "um_infer.py"]
    
    total_time = 0
    output = {}
    
    # 运行每个脚本并累积时间
    for script in scripts:
        exec_time, script_output = run_script(script)
        total_time += exec_time
        output[script] = script_output
    
    # 计算平均执行时间
    avg_time = total_time / len(scripts)
    
    print(f"\nAverage execution time for all scripts: {avg_time:.4f} seconds")
    
    return avg_time, output

# 执行所有脚本并返回结果
avg_time, output = run_all_scripts()

# 可以将输出保存到文件中
with open("execution_output.txt", "w") as f:
    f.write(f"Average execution time: {avg_time:.4f} seconds\n")
    f.write("\n--- Outputs from each script ---\n")
    for script, script_output in output.items():
        f.write(f"\nOutput of {script}:\n")
        f.write(script_output)
        f.write("\n" + "-"*50 + "\n")

print("Execution completed. Check 'execution_output.txt' for detailed outputs.")

