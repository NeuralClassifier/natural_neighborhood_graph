# Natural neighbor: A self-adaptive neighborhood method without parameter K
Authors: Qingsheng Zhu, Ji Feng, and Jinlong Huang

Published in Pattern Recognition Letters, Elsevier (2016)

## Abstract
K-nearest neighbor (KNN) and reverse k-nearest neighbor (RkNN) are two bases of many well-established and high-performance pattern-recognition techniques, but both of them are vulnerable to their parameter choice. Essentially, the challenge is to detect the neighborhood of various data sets, while utterly ignorant of the data characteristic. In this paper, a novel concept in terms of nearest neighbor is proposed and named natural neighbor (NaN). In contrast to KNN and RkNN, it is a scale-free neighbor, and it can reflect a better data characteristics. This article discusses the theoretical model and applications of natural neighbor in a different field, and we demonstrate the improvement of the proposed neighborhood on both synthetic and real-world data sets.

Paper Link: [click here](https://www.sciencedirect.com/science/article/pii/S016786551630085X?casa_token=EguvnJRZZ28AAAAA:3Dg5PuxkMUc5RWbw-vngaKwo_08fCaygbVL73DivuHk9s4VcxoEHB8kpAeGQFEgdjeHxXp0IjQ)

## Content

This repository contains the source code and real-life datasets.

  * `natural_neighborhood_g.py`: This python file gives only the source code for the natural neighborhood graph (NaNG).
  * `NaturalNeighborhood_G_implement.py`: This script implements the NaNG on real-life datasets
  * `df_to_consider`: This folder has eight different real-life datasets to test NaNG.
  
# Instructions

The program is written in Python 3.8:

* Using conda:
```
conda install -c conda-forge jupyterlab
```
* or using pip:
```
pip install jupyterlab
```
* Download Python: [Click Here](https://www.python.org/downloads/)

## Dependencies
The program requires the following Python libraries:
* numpy v1.21.5
* scikit-learn v1.0.1
* pandas v1.3.4
* scipy v1.7.3

# Contributors

* Kushankur Ghosh, [kushanku@ualberta.ca](mailto:kushanku@ualberta.ca)

