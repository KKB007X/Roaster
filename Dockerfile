
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    glslc \
    && rm -rf /var/lib/apt/lists/*
    

WORKDIR /app

COPY . .

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j2

ENTRYPOINT ["/app/build/roaster"]