#!/bin/bash
START=1
END=15

for (( run_id=$START; run_id<$END; run_id++ ))
do
    echo "========== Starting Run $run_id =========="

    # Run with a timeout of 20 minutes (1200 seconds)
    timeout 1200s python3 open_and_put_item_voxposer.py $run_id

    # Check exit status
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "[⏱️] Run $run_id timed out after 20 minutes. Moving on."
    elif [ $exit_code -ne 0 ]; then
        echo "[⚠️] Run $run_id failed with exit code $exit_code. Skipping to next."
    else
        echo "[✅] Run $run_id completed successfully."
    fi

    echo "Sleeping and resetting GPU memory (if needed)..."
    sleep 10
done
