#!/bin/bash

docker build -f docker/Dockerfile . -t hydra-moe:local

docker run --gpus all --env-file .env hydra-moe:local 
