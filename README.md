# UMMAN: Unsupervised Multi-graph Merge Adversarial Network for Disease Prediction Based on Intestinal Flora
Our paper has been accepted for publication in IEEE Transactions on Computational Biology and Bioinformatics.

This repository contains the original Python code for our paper [UMMAN: Unsupervised Multi-graph Merge Adversarial Network for Disease Prediction Based on Intestinal Flora](https://ieeexplore.ieee.org/abstract/document/10908075). The full code will be open-sourced in the future.

# Overview
Gut microbiome composition is closely linked to human diseases, but predicting diseases from OTU data is challenging due to complex microbial interactions. Existing methods fail to capture these associations across hosts, limiting performance.

We propose UMMAN (Unsupervised Multi-Graph Merge Adversarial Network), the first approach integrating Graph Neural Networks (GNNs) for gut microbiome disease prediction.

Experiments on five benchmark datasets demonstrate UMMAN’s effectiveness and stability, providing a scalable solution for microbiome-based disease prediction.

# Contributions
- We are the first to apply graph machine learning to disease prediction based on intestinal flora. We propose a novel framework Unsupervised Multi-graph Merge Adversarial Network (UMMAN), which learns the intricate connections among gut microbes of different hosts to guide disease prediction.
- The Original-Graph is constructed using multiple types of relational metrics, and a Shuffled-Graph is created by disrupting its node associations. Through the use of adversarial loss and hybrid attention loss, the embeddings are trained to closely align with the Original-Graph while diverging as much as possible from the Shuffled-Graph.
- We propose a Node Feature Global Integration (NFGI) descriptor, which includes both node-level and graph-level stages to better represent the global embedding of a graph.
- Experiments on the benchmark datasets demonstrate that our UMMAN achieves state-of-the-art performance in the disease prediction task of gut microbiota, and also prove that our method is more stable than previous approaches.

# Proposed Framework
We propose UMMAN, which constructs an Original-Graph with multiple relations and a Shuffled-Graph for contrastive learning. It leverages a Node Feature Global Integration (NFGI) module and a joint adversarial-hybrid attention loss to enhance representation learning.
![Figure1](https://github.com/Dingkun0817/UMMAN/blob/main/Figures/UMMAN.jpg)

Figure 1: (a) Overview of the proposed UMMAN architecture. The Original-Graph and Shuffled-Graph are processed through GCN and fused using an attention block, and the NFGI module captures the global features of the graph. (b) NFGI module with two stages: Node-level and Graph-level. The adversarial comparison between the Original-Graph and Shuffled-Graph is conducted with a joint loss function to enhance the authenticity of learned relationships.

# Key Results
- Better Performance: As shown in Figure 2, UMMAN significantly improves disease classification performance on five datasets, surpassing traditional machine learning and deep learning methods.

![Figure2](https://github.com/Dingkun0817/UMMAN/blob/main/Figures/Figure3.png)

Figure 2: Intuitive comparison of our method with previous work on the five datasets.

- Enhanced Cross-Host Generalization: As shown in Figure 3, UMMAN strengthens the similarity among hosts within the same class while increasing the dissimilarity between different host classes.

![Figure3](https://github.com/Dingkun0817/UMMAN/blob/main/Figures/Figure4.png)
Figure 3: Graph representation of host correlation before and after UMMAN.

- Stable Performance Across Datasets: As shown in Figure 4, the model performs well on five OTU datasets across a range of weight values, demonstrating its robustness and reliability.
![Figure4_1](https://github.com/Dingkun0817/UMMAN/blob/main/Figures/Figure5_Cirrhosis_Acc.png)
