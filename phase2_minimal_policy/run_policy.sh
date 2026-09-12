#!/bin/sh
set -e
. /opt/ros/jazzy/setup.sh
set -u
exec python3 /policy/policy.py "$@"
