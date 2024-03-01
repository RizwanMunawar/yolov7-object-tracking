#!/bin/bash

# Create or clear the file where the output will be stored
> output.log

# Infinite loop
while true; do
  echo "$(date) - Still running..." >> output.log
  sleep 5
done
