# Uses GPU enabled torch
# FROM public.ecr.aws/artefacts/go2:mujoco
# Uses CPU only torch
FROM public.ecr.aws/artefacts/go2:mujoco-cputorch
WORKDIR /ws

COPY artefacts.yaml go2_wtw_demo.py utils.py ./
COPY tests/ tests/
COPY resources/ resources/

CMD artefacts run $ARTEFACTS_JOB_NAME
