# Multi-Agent-Unity
Distributed RL algorithms developed on Multi-Agent gym wrapper for unity environment

# Environment
You can use either conda or docker

## docker
```shell
cd docker
bash build.sh
docker run -it -v ${path to the code in your computer}:/unity unity:1.0 bash
cd /unity
python training.py
```

## conda
``` shell
conda env create -f environment.yml
conda activate unity
python training.py
```