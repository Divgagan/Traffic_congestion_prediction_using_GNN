<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Traffic%20Congestion%20Prediction&fontSize=30&fontColor=fff&animation=twinkling&desc=Using%20Graph%20Neural%20Networks&descSize=16&descAlignY=75" width="100%"/>

# ðŸš¦ Traffic Congestion Prediction using GNN

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Stars](https://img.shields.io/github/stars/Divgagan/Traffic_congestion_prediction_using_GNN?style=for-the-badge&color=yellow)](https://github.com/Divgagan/Traffic_congestion_prediction_using_GNN/stargazers)

</div>

## ðŸ“Œ Overview

This project implements a **Graph Neural Network (GNN)** model to predict traffic congestion in urban road networks. By modeling road intersections as graph nodes and roads as edges, the model captures spatial dependencies between traffic signals to make accurate congestion predictions.

## ðŸ§  Architecture

`
Road Network Graph
       â”‚
  Node Features â”€â”€â–º Graph Convolutional Layers â”€â”€â–º Prediction Head
  Edge Features â”€â”€â–º          (GCN/GAT)             (Congestion Level)
`

## âœ¨ Features

- ðŸ“Š Graph-based traffic network modeling
- ðŸ” Temporal + spatial feature extraction
- âš¡ Efficient training on large road graphs
- ðŸ“ˆ Visualization of congestion heatmaps

## ðŸ› ï¸ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | PyTorch / PyTorch Geometric |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, NetworkX |

## ðŸš€ Getting Started

`ash
# Clone the repo
git clone https://github.com/Divgagan/Traffic_congestion_prediction_using_GNN.git
cd Traffic_congestion_prediction_using_GNN

# Install dependencies
pip install -r requirements.txt

# Run the model
python train.py
`

## ðŸ“‚ Project Structure

`
â”œâ”€â”€ data/               # Traffic dataset
â”œâ”€â”€ models/             # GNN model definitions
â”œâ”€â”€ utils/              # Helper functions
â”œâ”€â”€ train.py            # Training script
â”œâ”€â”€ evaluate.py         # Evaluation script
â””â”€â”€ requirements.txt
`

## ðŸ‘¤ Author

**Gagan Diwakar** â€” [Portfolio](https://portfolio-gagan-nu.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/gagan-diwakar-772134293/) | [GitHub](https://github.com/Divgagan)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>