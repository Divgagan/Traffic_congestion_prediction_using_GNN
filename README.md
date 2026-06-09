<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Traffic%20Congestion%20Prediction&fontSize=30&fontColor=fff&animation=twinkling&desc=Graph%20Neural%20Networks&descSize=16&descAlignY=75" width="100%"/>

<div align="center">

# Traffic Congestion Prediction using GNN

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pyg.org)
[![Stars](https://img.shields.io/github/stars/Divgagan/Traffic_congestion_prediction_using_GNN?style=for-the-badge&color=yellow)](https://github.com/Divgagan/Traffic_congestion_prediction_using_GNN/stargazers)

</div>

---

## Overview

This project implements a **Graph Neural Network (GNN)** model to predict traffic congestion in urban road networks. Road intersections are modeled as graph nodes and roads as edges, allowing the model to capture spatial dependencies between traffic signals.

---

## Architecture

```
Road Network Graph
       |
  Node Features --> Graph Convolutional Layers --> Prediction Head
  Edge Features -->        (GCN / GAT)            (Congestion Level)
```

---

## Features

- Graph-based traffic network modeling
- Temporal and spatial feature extraction
- Efficient training on large road graphs
- Visualization of congestion heatmaps

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | PyTorch + PyTorch Geometric |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, NetworkX |

---

## Getting Started

```bash
git clone https://github.com/Divgagan/Traffic_congestion_prediction_using_GNN.git
cd Traffic_congestion_prediction_using_GNN
pip install -r requirements.txt
python train.py
```

---

## Project Structure

```
.
|-- data/               # Traffic dataset
|-- models/             # GNN model definitions
|-- utils/              # Helper functions
|-- train.py            # Training script
|-- evaluate.py         # Evaluation script
|-- requirements.txt
|-- .gitignore
```

---

**Author:** [Gagan Diwakar](https://portfolio-gagan-nu.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/gagan-diwakar-772134293/) | [GitHub](https://github.com/Divgagan)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>