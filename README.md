# Embodied Intelligence via Learning and Evolution

This is the code for the paper

**<a href="https://www.nature.com/articles/s41467-021-25874-z">Embodied Intelligence via Learning and Evolution</a>**
<br>
<a href="http://web.stanford.edu/~agrim/">Agrim Gupta</a>,
<a href="http://cvgl.stanford.edu/silvio/">Silvio Savarese</a>,
<a href="https://ganguli-gang.stanford.edu/surya.html">Surya Ganguli</a>,
<a href="http://vision.stanford.edu/feifeili/">Fei-Fei Li</a>
<br>

The intertwined processes of learning and evolution in complex environmental niches have resulted in a remarkable diversity of morphological forms. Moreover, many aspects of animal intelligence are deeply embodied in these evolved morphologies. However, the principles governing relations between environmental complexity, evolved morphology, and the learnability of intelligent control, remain elusive, partially due to the substantial challenge of performing large-scale in silico experiments on evolution and learning. We introduce Deep Evolutionary Reinforcement Learning (DERL): a novel computational framework which can evolve diverse agent morphologies to learn challenging locomotion and manipulation tasks in complex environments using only low level egocentric sensory information. Leveraging DERL we demonstrate several relations between environmental complexity, morphological intelligence and the learnability of control.

<div align='center'>
<img src="images/teaser.gif"></img>
</div>

## Code Structure

The code consists of three main components:

1. [UNIMAL Design Space](derl/envs/morphology.py): A UNIversal aniMAL morphological design space that is both highly expressive yet also enriched for useful controllable morphologies.
2. [DERL](tools/evolution.py): An efficient asynchronous method for parallelizing computations underlying learning and evolution across many compute nodes.
3. [Evolutionary environments and evaluation tasks](derl/envs):  A set of three evolutionary environments and eight evaluation tasks. 

## Setup
We provide [Dockerfile](docker/Dockerfile) for easy installation and development. If you prefer to work without docker please take a look at Dockerfile and ensure that your local system has all the necessary dependencies installed. 

## Evolving Unimals
```bash
# Build docker container. Ensure that MuJoCo license is present: docker/mjkey.txt
./scripts/build_docker.sh
# Evolve unimals. Please change MOUNT_DIR location inside run_docker_cpu.sh
./scripts/run_docker_cpu.sh python tools/evolution.py --cfg ./configs/evo/ft_test.yml NODE_ID 0
```

The default parameters assume that you are running the code on 16 machines. Please ensure that each machine has a minimum of 72 CPUs. While running the script on multiple nodes you would have to ensure that NODE_ID on each machine is unique and between [0, NUM_NODES - 1].

## Visualizing Environments
If you have installed all dependencies in your local machine. You can visualize the environment as follows:

```bash
python tools/terrain_builder.py --cfg configs/evo/mvt.yml
```

## Citation
If you find this code useful, please consider citing:

```text
@article{gupta2021embodied,
  title={Embodied intelligence via learning and evolution},
  author={Gupta, Agrim and Savarese, Silvio and Ganguli, Surya and Fei-Fei, Li},
  journal={Nature communications},
  volume={12},
  number={1},
  pages={5721},
  year={2021},
  publisher={Nature Publishing Group}
}

```

## Credit

This codebase would not have been possible without the following amazing open source codebases:

1. [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
2. [hill-a/stable-baselines](https://github.com/hill-a/stable-baselines)
3. [deepmind/dm_control](https://github.com/deepmind/dm_control)
4. [openai/multi-agent-emergence-environments](https://github.com/openai/multi-agent-emergence-environments)
