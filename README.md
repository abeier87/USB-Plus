# USB-Plus

As a benchmark for Semi-Supervised Learning (SSL) and Long-Tailed SSL (LTSSL) algorithms, [USB](https://github.com/microsoft/Semi-supervised-learning) provides a fair and convenient platform for comparing different learning algorithms.

Considering that the official repository of USB has not been updated for a year, we have collected some of the latest algorithms that can already run directly on the USB framework but have not been officially included by USB. For details, please see [1](#1-latest-ssl-algorithms) and [2](#2-latest-ltssl-algorithms).

For those SSL/LTSSL algorithms that cannot be run directly on USB, we have migrated them to the USB framework. The usage instructions are shown in [3](#3-how-to-use).

## 1. Latest SSL Algorithms
1. Qian Shao *et. al.*, Enhancing Semi-Supervised Learning via Representative and Diverse Sample Selection, NeurIPS 2024, [Code Link](https://github.com/YanhuiAILab/RDSS).


## 2. Latest LTSSL Algorithms
1. Chengcheng Ma *et. al.*, Three Heads Are Better than One: Complementary Experts for Long-Tailed Semi-supervised Learning, AAAI 2024, [Code Link](https://github.com/machengcheng2016/CPE-LTSSL).

2. Tong Wei *et. al.*, Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need, CVPR 2023, [Code Link](https://github.com/machengcheng2016/CPE-LTSSL).

3. Yue Duan *et. al.*, RDA: Reciprocal Distribution Alignment for Robust Semi-supervised Learning, ECCV 2022, [Code Link](https://github.com/machengcheng2016/CPE-LTSSL).

## 3. How to Use
Each folder contains one algorithm.  
If it is an SSL algorithm, copy it to the USB/semilearn/algorithms directory;  
If it is an LTSSL algorithm, copy it to the USB/semilearn/imb_algorithms directory.  
Then, all you need to do is register the algorithm in USB/semilearn/core/utils/registry.py.