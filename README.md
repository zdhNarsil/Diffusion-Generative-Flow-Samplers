# Diffusion Generative Flow Samplers 

Diffusion Generative Flow Samplers (DGFS):
Improving learning signals through partial trajectory optimization

[Dinghuai Zhang](https://zdhnarsil.github.io/), 
[Ricky Tian Qi Chen](https://rtqichen.github.io//),
[Cheng-Hao Liu](https://pchliu.github.io/), 
Aaron Courville, 
[Yoshua Bengio](https://yoshuabengio.org/).

We propose a novel DGFS sampler for continuous space sampling 
from given unnormalized densities based on stochastic optimal control 🤖 formulation
and the probabilistic 🎲 GFlowNet framework.

<a href="https://imgse.com/i/pPOmv7T"><img src="https://z1.ax1x.com/2023/10/03/pPOmv7T.md.png" alt="pPOmv7T.png" border="0" /></a>

`target/` has the target distribution code.
`gflownet/` contains the DGFS algorithm code.

## Examples

```bash
python -m gflownet.main target=gm dt=0.05
python -m gflownet.main target=funnel
python -m gflownet.main target=wells
```

## Dependency

Apart from commonly used torch, torchvision, numpy, scipy, matplotlib,
we use the following packages:
```bash
pip install hydra-core omegaconf submitit hydra-submitit-launcher
pip install wandb tqdm einops seaborn ipdb
```
