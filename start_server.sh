#!/bin/bash
cd "/mnt/c/Users/Mohamed Ahmed/Desktop/mini-rag/src"
/home/mohamed_ahmed/miniconda3/envs/mini-rag/bin/python -m uvicorn main:app --host 0.0.0.0 --port 5000
