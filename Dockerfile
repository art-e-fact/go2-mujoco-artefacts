# Uses GPU enabled torch
# FROM public.ecr.aws/artefacts/go2:mujoco
# Uses CPU only torch
FROM public.ecr.aws/artefacts/go2:mujoco-cputorch
WORKDIR /ws

COPY artefacts.yaml go2_wtw_demo.py utils.py ./
COPY tests/ tests/ 
COPY resources/scene_flat.xml src/unitree_mujoco/unitree_robots/go2/

CMD artefacts run $ARTEFACTS_JOB_NAME
