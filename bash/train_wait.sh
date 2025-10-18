#!/bin/bash

# --- CONFIGURATION ---
# The PID of the training process you are currently waiting for.
PID_TO_WATCH=3893417 
# How many seconds to wait between checks.
CHECK_INTERVAL=600 

echo "Waiting for the previous training process (PID: $PID_TO_WATCH) to complete."
echo "Checking every $CHECK_INTERVAL seconds..."

# This loop will continue as long as the process with that PID exists.
# 'kill -0' is a special command that doesn't actually kill the process.
# It just checks if the process exists and if you have permission to signal it.
# It succeeds (returns 0) if the process is running, and fails (returns non-zero) if it's not.
while kill -0 $PID_TO_WATCH >/dev/null 2>&1; do
    echo "[$(date)] Process $PID_TO_WATCH is still running. Waiting..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "--------------------------------------------------------"
echo "Success! The previous training process (PID: $PID_TO_WATCH) has completed."
echo "Starting the new training now."
echo "--------------------------------------------------------"
echo ""

# --- YOUR NEW COMMAND ---
# Now, you can launch your new deepspeed training.
nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_SHAPES_RANDOM.toml' > ./output/bokeh_nohup_log/3dshape_random_bokeh.out 2>&1 &

# Optional: Get the PID of the NEW process you just started
NEW_PROCESS_ID=$!
echo "New training has been started in the background with PID: $NEW_PROCESS_ID"