# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install all required libs
RUN apt-get update && \
    apt-get install -y \
    bc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY detect.py .
COPY pexels_videos_2670(1080p).mp4 .

# Copy the current directory contents and run script into the container
COPY . .
COPY run.sh .
COPY infinite_loop.sh .

# Make the startup script executable
RUN chmod +x run.sh
RUN chmod +x infinite_loop.sh

# Run the startup script when the container launches
CMD ["./run.sh"]
#CMD ["./infinite_loop.sh"]

#For debugg
#CMD ["/bin/bash"]
