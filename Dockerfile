FROM ros:jazzy-ros-base-noble

LABEL org.opencontainers.image.source="https://github.com/BigCatwanzi20071202/ebim-task1-mujoco-submission"
LABEL org.opencontainers.image.description="EBiM Task 1 Phase II minimal safe-hold policy"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    ROS_DOMAIN_ID=0

WORKDIR /policy
COPY phase2_minimal_policy/ /policy/
RUN useradd --create-home --uid 10001 policy \
 && chmod 0555 /policy/run_policy.sh /policy/policy.py /policy/policy_core.py

USER policy
ENTRYPOINT ["/policy/run_policy.sh"]
CMD ["run"]
