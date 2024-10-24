# going to cuda image 12/23
# transcription stream ts-gpu image
#
# Start from a smaller base image, such as Ubuntu slim or Alpine if possible
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04


# Set environment variables to reduce interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Declare the argument to accept the password during the build
ARG TRANSCRIPTION_USER_PASSWORD

# Update and install necessary packages in a single RUN command and clean up in the same layer
RUN apt-get update && apt-get install -y \
    python3.9 \
    ssh \
    openssh-server \
    ffmpeg \
    python3-pip \
    build-essential \
    cython3 \
    git \
    python3-wheel \
    dos2unix \
    TZDATA \
 && rm -rf /var/lib/apt/lists/*

# Added to fix sox/numpy error while installing whisper-diarization 03/24/2024
RUN pip install typing_extensions
RUN pip install numpy

# Clone only the necessary branch or commit of whisper-diarization to reduce size
RUN git clone --depth 1 https://github.com/transcriptionstream/whisper-diarization.git

# Install whisper-diarization requirements
WORKDIR /whisper-diarization
RUN pip install huggingface-hub==0.23.2
RUN pip install -r requirements.txt

# Install req for calls to ts-meilisearch
RUN pip install meilisearch

# Copy the required files
COPY ts-control.sh /root/scripts/
COPY transcribe_example_d.sh /root/scripts/transcribe_example.sh
COPY test.wav /home/transcriptionstream/
COPY ts-summarize.py /root/scripts/
COPY index-single.py /root/scripts/
COPY auto-summary.py /root/scripts/

# Run dos2unix on the copied scripts to fix line endings
RUN dos2unix /root/scripts/ts-control.sh \
    && dos2unix /root/scripts/transcribe_example.sh

# Create a new user and setup the environment
RUN useradd -m -p $(openssl passwd -1 "$TRANSCRIPTION_USER_PASSWORD") transcriptionstream \
 && mkdir -p /home/transcriptionstream/incoming \
 && mkdir -p /home/transcriptionstream/transcribed \
 && mkdir -p /home/transcriptionstream/incoming/diarize \
 && mkdir -p /home/transcriptionstream/incoming/transcribe \
 && chown -R transcriptionstream:transcriptionstream /home/transcriptionstream/ \
 && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config

# Run the necessary scripts so we have our transcription models IN the image. Adds to build time for download.
RUN python3 diarize.py -a /home/transcriptionstream/test.wav
RUN whisperx --model large-v3 --language en /home/transcriptionstream/test.wav --compute_type int8
#RUN python3 -m pytorch_lightning.utilities.upgrade_checkpoint ../root/.cache/torch/whisperx-vad-segmentation.bin

# Expose the SSH port
EXPOSE 22

# Start ssh and run the control script
CMD service ssh start && while true; do bash /root/scripts/ts-control.sh; sleep 5; done
