#!bin/bash

timeout 7h nvidia-smi -l 5 -f gpu_test.log --query-gpu=timestamp,utilization.gpu,clocks.sm,pstate,temperature.gpu,fan.speed,memory.used --format=csv