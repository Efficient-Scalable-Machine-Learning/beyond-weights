# Beyond Weights
This repository contains an implementation of the paper [Beyond Weights: Deep learning in Spiking Neural Networks with pure synaptic-delay training](https://arxiv.org/abs/2306.06237).

[These mastodon posts](https://sigmoid.social/@anandsubramoney/110819773391545444) provide a short summary of this work.

This code is a forked version of the [SLAYER](https://github.com/bamsumit/slayerPytorch) repository with appropriate changes.

# Usage #
The `environment.yml` file provides a snapshot of the conda environment that can be used with `conda env create -f
environment.yml`. 
Install this package with `python setup.py install`
Note that you might have to install the pip packages listed there manually.

Experiments with MNIST and Fashion MNIST are on ./example/experiments, to run simply "python file_name.py"


## Citation ##
Edoardo W. Grappolini and Anand Subramoney. Beyond weights: Deep learning in spiking neural networks with pure synaptic-delay training. In International Conference on Neuromorphic Systems (ICONS ’23), Santa Fe, NM, USA. ACM, June 2023.

```bibtex
@inproceedings{grappolini2023weights,
      title={Beyond Weights: Deep learning in Spiking Neural Networks with pure synaptic-delay training}, 
      author={Edoardo W. Grappolini and \dotuline{Anand Subramoney}},
      booktitle = {International Conference on Neuromorphic Systems (ICONS '23), Santa Fe, NM, USA},
      year={2023},
      month = jun,
      eprint={2306.06237},
      primaryClass={cs.NE},
    publisher = {{ACM}},
    preprint = {https://arxiv.org/abs/2306.06237},
}
```
