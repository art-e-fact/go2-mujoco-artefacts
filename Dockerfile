FROM tomolnorman/go2-mujoco-artefacts:latest

WORKDIR /ws

COPY artefacts.yaml go2_wtw_demo.py ./
COPY tests/ tests/ 
COPY resources/scene_flat.xml src/unitree_mujoco/unitree_robots/go2/

CMD artefacts run $ARTEFACTS_JOB_NAME
