# Use Debian 12 as the base image
FROM debian:12

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    unzip \
    wget \
    git \
    build-essential \
    libglib2.0-0 \
    libnss3 \
    libxss1 \
    libjpeg-dev zlib1g-dev \
    libasound2 \
    fonts-liberation \
    libappindicator3-1 \
    libx11-xcb1 \
    libdbus-glib-1-2 \
    xdg-utils \
    chromium \
    chromium-driver \
    && apt-get clean

# Install Python packages
RUN pip install --upgrade pip --break-system-packages && \
    pip install streamlit \
    farm-haystack[colab,inference] farm-haystack[preprocessing]  --break-system-packages





# Pre-download models (store in local directory)
RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('deepset/roberta-base-squad2')" && \
    python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('t5-base')" && \
    python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')" && \
    python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('t5-base')"

# Set the working directory
WORKDIR /usr/src/app

# Copy the script and any necessary files into the container
COPY . .

RUN mkdir -p "data/build_your_first_question_answering_system"

RUN cp doc.txt data/build_your_first_question_answering_system/doc.txt

# Add a path for the pre-downloaded models (this ensures they're accessible offline)
ENV TRANSFORMERS_CACHE=/usr/src/app/.cache

# Command to run the script
CMD ["streamlit","run", "sss.py"]
