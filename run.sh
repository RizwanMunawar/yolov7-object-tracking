#!/bin/bash

# Number of times to run the detection
num_runs=10

# Initialize total time
total_time=0

for ((i=1; i<=$num_runs; i++)); do
    echo "Run $i:"
    
    # Record the start time
    start_time=$(date +"%Y-%m-%d %H:%M:%S.%N")

    # Run the Python command and store the output in output.txt
    { time python detect.py --weights yolov7.pt --source "pexels_videos_2670(1080p).mp4"; } 2>> output.txt

    # Record the end time
    end_time=$(date +"%Y-%m-%d %H:%M:%S.%N")

    # Calculate the time taken
    start_seconds=$(date -d "$start_time" +"%s.%N")
    end_seconds=$(date -d "$end_time" +"%s.%N")
    time_taken=$(echo "$end_seconds - $start_seconds" | bc)

    # Display the time taken for each run
    echo "Time taken for Run $i: $time_taken seconds"

    # Accumulate the total time
    total_time=$(echo "$total_time + $time_taken" | bc)
done

# Calculate the average time
average_time=$(echo "$total_time / $num_runs" | bc)

# Store the average time in a file
echo "Average Time taken for $num_runs runs: $average_time seconds" >> output.txt

# Display the average time
echo "Average Time taken for $num_runs runs: $average_time seconds"

