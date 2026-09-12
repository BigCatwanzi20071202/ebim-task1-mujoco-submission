#!/bin/sh
set -eu
. /opt/ros/jazzy/setup.sh
exec python3 /policy/policy.py "$@"
