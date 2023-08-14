<h1> Exploring Sparsity in Graph Transformers </h1>

Open-sourced implementation for AAAI 2024 Submission (Paper ID: 4898).


<h2> Abstract </h2>

Graph Transformers (GTs) have achieved impressive results on various graph-related tasks. However, the huge computational cost of GTs hinders their deployment and application, especially in resource-constrained environments. Therefore, in this paper, we explore the feasibility of sparsifying GTs, a significant yet under-explored topic.

We first discuss the redundancy of GTs based on the characteristics of existing GT models, then propose a comprehensive Graph Transformer SParsification (GTSP) framework that helps to reduce the computational complexity of GTs from four dimensions: the input graph data, attention heads, model layers, and model weights.

We examine our GTSP through extensive experiments on prominent GTs, including GraphTrans, Graphormer, and GraphGPS. The experimental results substantiate that GTSP effectively cuts computational costs, accompanied by only marginal decreases in accuracy or, in some cases, even improvements. For instance, GTSP yields a reduction of 30% in Floating Point Operations while contributing to a 1.8% increase in Area Under the Curve accuracy.

Furthermore, we provide several insights on the characteristics of attention heads and the behavior of attention mechanisms, all of which have immense potential to inspire future research endeavors in this domain.



<h2> Python Dependencies </h2>

Our proposed Gapformer is implemented in Python 3.7 and major libraries include:

* [Pytorch](https://pytorch.org/) = 1.11.0+cu113
* [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) torch-geometric=2.2.0

More dependencies are provided in **requirements.txt**.

<h2> To Run </h2>

Once the requirements are fulfilled, use this command to run Gapformer:

`sh test.sh`

<h2> Datasets </h2>

All datasets used in this paper can be downloaded from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).

<h2> Baselines </h2>

All GCN-based and heterophily-based methods are implemented based on [PyG](https://github.com/pyg-team/pytorch_geometric)


Graph Transformer baselines and their code URLs are:

* GraphTrans: https://github.com/ucbrise/graphtrans

* Graphormer: https://github.com/Microsoft/Graphormer

* GraphGPS: https://github.com/rampasek/GraphGPS



