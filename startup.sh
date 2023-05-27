#!/bin/bash

# Stop existing notebook server
jupyter notebook stop 8888 &&
jupyter notebook --port=8888 &
when-changed requirements_dev.in pip-compile requirements_dev.in &
when-changed requirements_dev.txt pip install -r requirements_dev.txt
