# UMMAN: Unsupervised Multi-graph Merge Adversarial Network for Disease Prediction Based on Intestinal Flora
Our paper has been accepted for publication in IEEE Transactions on Computational Biology and Bioinformatics.

This repository contains the original Python code for our paper [UMMAN: Unsupervised Multi-graph Merge Adversarial Network for Disease Prediction Based on Intestinal Flora](https://ieeexplore.ieee.org/abstract/document/10908075). 

# Overview
The abundance of intestinal flora is closely related to human diseases, but diseases are not caused by a single gut microbe. Instead, they result from the complex interplay of numerous microbial entities. This intricate and implicit connection among gut microbes poses a significant challenge for disease prediction using abundance information from OTU data.

We introduce Unsupervised Multi-Graph Merge Adversarial Network (UMMAN), a novel architecture designed to address this challenge. UMMAN can obtain the embeddings of nodes in the Multi-Graph with an unsupervised scenario, effectively learning the multiplex and implicit associations. It is the first approach to integrate Graph Neural Networks (GNNs) for intestinal flora disease prediction.

UMMAN constructs an Original-Graph using multiple relation types and generates a Shuffled-Graph by disrupting the nodes. The model leverages the Node Feature Global Integration (NFGI) module to capture global graph features and employs a joint loss function that combines adversarial loss and hybrid attention loss. This ensures that the real graph embedding aligns closely with the Original-Graph and diverges from the Shuffled-Graph.

Experiments on five benchmark OTU gut microbiome datasets demonstrate the effectiveness and stability of UMMAN, providing a scalable and robust solution for microbiome-based disease prediction.

# Contributions
- We are the first to apply graph machine learning to disease prediction based on intestinal flora. We propose a novel framework Unsupervised Multi-graph Merge Adversarial Network (UMMAN), which learns the intricate connections among gut microbes of different hosts to guide disease prediction.
- The Original-Graph is constructed using multiple types of relational metrics, and a Shuffled-Graph is created by disrupting its node associations. Through the use of adversarial loss and hybrid attention loss, the embeddings are trained to closely align with the Original-Graph while diverging as much as possible from the Shuffled-Graph.
- We propose a Node Feature Global Integration (NFGI) descriptor, which includes both node-level and graph-level stages to better represent the global embedding of a graph.
- Experiments on the benchmark datasets demonstrate that our UMMAN achieves state-of-the-art performance in the disease prediction task of gut microbiota, and also prove that our method is more stable than previous approaches.

# Proposed Framework
The architecture of UMMAN is shown in Figure 1, where nodes represent hosts and multiplex indicators are used to construct the Original-Graph based on the similarity between nodes. To enhance the learning of associations, a Shuffled-Graph is introduced by disrupting these relationships. Both graphs are updated by Graph Convolutional Network (GCN), and node embeddings are generated through an attention block. 

The Node Feature Global Integration (NFGI) descriptor then aggregates these embeddings into a global graph representation. To capture complex relationships among gut microbes across hosts while ensuring alignment with the Original-Graph and divergence from the Shuffled-Graph, we propose a joint loss function that combines adversarial loss and hybrid attention loss.
<div align=center>
<img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/UMMAN-1.png" alt="Figure1_1" width="420"><img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/UMMAN-2.png" alt="Figure1_2" width="420">
</div>
<div align=center>
<img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/UMMAN-1.png" alt="Figure1">
</div>
Figure 1: (a) Overview of the proposed UMMAN architecture. The Original-Graph and Shuffled-Graph are processed through GCN and fused using an attention block, and the NFGI module captures the global features of the graph. (b) NFGI module with two stages: Node-level and Graph-level. The adversarial comparison between the Original-Graph and Shuffled-Graph is conducted with a joint loss function to enhance the authenticity of learned relationships.

# Key Results
- Better Performance: As shown in Figure 2, UMMAN significantly improves disease classification performance on five datasets, surpassing traditional machine learning and deep learning methods.
<div align=center>
<img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure3-1.png" alt="Figure3_1" width="420"><img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure3-2.png" alt="Figure3_2" width="420">
</div>
<p align="center">
Figure 2: Intuitive comparison of our method with previous work on the five datasets.
</p>

- Enhanced Cross-Host Generalization: As shown in Figure 3, UMMAN strengthens the similarity among hosts within the same class while increasing the dissimilarity between different host classes.
<div align=center>
<img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure4-1.png" alt="Figure3" width="460">
</div>
<p align="center">
Figure 3: Graph representation of host correlation before and after UMMAN.
</p>

- Stable Performance Across Datasets: As shown in Figure 4, the model performs well on five OTU datasets across a range of weight values, demonstrating its robustness and reliability.
<div align=center>
<img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure5_Cirrhosis_Acc.png" alt="Figure4_1" width="250"><img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure5_IBD_AUC.png" alt="Figure4_2" width="250"><img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure5_Obesity_AUC.png" alt="Figure4_3" width="250">
</div>
<div align=center>
<img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure5_T2D_Acc.png" alt="Figure4_4" width="250"><img src="https://raw.githubusercontent.com/Dingkun0817/UMMAN/main/Figures/Figure5_WT2D_Acc.png" alt="Figure4_5" width="250">
</div>
<p align="center">
Figure 4: Hyperparameter selection for adversarial loss and hybrid attention loss.
</p>

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{liu2025umman,
  title={Umman: Unsupervised multi-graph merge adversarial network for disease prediction based on intestinal flora},
  author={Liu, Dingkun and Zhou, Hongjie and Qu, Yilu and Zhang, Huimei and Xu, Yongdong},
  journal={IEEE Transactions on Computational Biology and Bioinformatics},
  year={2025},
  publisher={IEEE}
}
```

## Contact
For any questions or collaborations, please feel free to reach out via d202481536@hust.edu.cn or open an issue in this repository.
