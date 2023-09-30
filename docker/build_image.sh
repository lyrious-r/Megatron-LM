#!/bin/bash
DOCKER_BUILDKIT=1 nvidia-docker build -f Dockerfile -t dynapipe:latest .