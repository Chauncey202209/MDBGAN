import re
import subprocess

import psutil
import datetime
import os
from pynvml import *

def linux_monitor(time):
    process = os.popen(f'nvidia-smi | grep {os.getpid()}')
    process_info = process.read()
    gpu_use=re.findall(r'\b(\d+)MiB\b',process_info)
    gpu_use=int(gpu_use[0])
    p = psutil.Process(int(os.getpid()))
    p.cpu_affinity()
    p.memory_full_info()
    command = f"top -b -n 1 -p {os.getpid()} | grep {os.getpid()}"
    output = subprocess.check_output(command, shell=True, encoding='utf-8')
    cpu_use = float(re.findall(r'\d+\.\d', output)[4])
    mem_percent = p.memory_percent()
    cpu_use=sum(psutil.cpu_percent(percpu=True))/24
    # 获取当前系统时间
    current_time = datetime.datetime.now().strftime("%F %T")  # %F年月日 %T时分秒
    cpu_rate = round(cpu_use , 3)
    menory_rate = round(mem_percent, 3)
    #import ipdb;ipdb.set_trace()
    gpu_rate = round( gpu_use/10015*100, 3)
    line = str(current_time) + ',' + str(cpu_rate) + ',' + str(menory_rate)+ ',' + str(gpu_rate)
    return line











    

