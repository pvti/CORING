
<p align="center" width="100%">
    <img src="assets\ring.png" width="20%" height="20%">
</p>

<div align="center">
    <em>
        "One model to compress them all, one approach to refine efficiency,<br>
        One method to decompose tensors and enhance neural proficiency.<br>
        In the realm of filters, CORING stands tall and true,<br>
        Preserving dimensions, accuracy it will accrue.<br>
        Experiments demonstrate its prowess, architectures put to test,<br>
        FLOPS and parameters reduced, accuracy manifest.<br>
        Like ResNet-50 in ImageNet's vast domain,<br>
        Memory and computation requirements it does restrain.<br>
        Efficiency elevated, generalization takes its flight,<br>
        In the world of neural networks, C:ring:RING shines its light."
    </em>
</div>


-----------------
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fpvtien96%2FSNRPruning&countColor=%23263759)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# :ring: Efficient tensor-based filter pruning
<div>
<div align="center">
    <a href='https://github.com/pvtien96' target='_blank'>Van Tien PHAM<sup>1,&#x2709</sup></a>&emsp;
    <a href='https://yzniyed.blogspot.com/p/about-me.html' target='_blank'>Yassine ZNIYED<sup>1</sup></a>&emsp;
    <a href='http://tpnguyen.univ-tln.fr/' target='_blank'>Thanh Phuong NGUYEN<sup>1</sup></a>&emsp;
</div>
<div>

<div align="center">
    <sup>1</sup>Université de Toulon, Aix Marseille Université, CNRS, LIS, UMR 7020, France&emsp;
    <sup>&#x2709</sup> Corresponding Author
</div>
<div style="text-align: justify"> We present a novel filter pruning method for neural networks, named CORING, for effiCient tensOr decomposition-based filteR prunING. The proposed approach preserves the multidimensional nature of filters by employing tensor decomposition. Our approach leads to a more efficient and accurate way to measure the similarity, compared to traditional methods that use vectorized or matricized versions of filters. This results in more efficient filter pruning without losing valuable information. Experiments conducted on various architectures proved its effectiveness. Particularly, the numerical results show that CORING outperforms state-of-the-art methods in terms of FLOPS and parameters reduction, and validation accuracy. Moreover, CORING demonstrates its ability to increase model generalization by boosting accuracy on several experiments. For example, with VGG-16, we achieve a 58.1% FLOPS reduction by removing 81.6% of the parameters, while increasing the accuracy by 0.46% on CIFAR-10. Even on the large scale ImageNet, for ResNet-50, the top-1 accuracy increased by 0.63%, while reducing 40.8% and 44.8% of memory and computation requirements, respectively. </div>

<div>
  <img class="image" src="assets\Framework.png" width="100%" height="100%">
</div>
<div align="center ">
    The CORING approach for filter pruning in one layer.
</div>


# :star2: News
Project is under development :construction_worker:. Please stay tuned for more :fire: updates.
* **2023.8.14:** [Poster](assets/poster.pdf) :bar_chart: is released. Part of the project will be :mega: presented at [GRETSI'23](https://gretsi.fr/colloque2023/) :clap:.
* **2023.8.12:** Baseline and compressed checkpoints :gift: are released.


# :dart: Main results 
<div align="center ">
  <img class="image" src="assets\Performance.png" width="40%" height="100%">
</div>
<div align="center ">
     Comparison of pruning methods for VGG-16 on CIFAR-10.
</div>

<div style="text-align: justify"> CORING is evaluated on  various benchmark datasets with well-known and representative architectures including the classic plain structure VGG-16-BN, the GoogLeNet with inception modules, the ResNet-56 with residual blocks, the DenseNet-40 with dense blocks and the MobileNetV2 with inverted residuals and linear bottlenecks. Due to a large number of simulations, these models are all considered on CIFAR-10. Also, to validate the scalability of CORING, we conduct experiments on the challenging ImageNet dataset with ResNet-50. </div>
 

<details>
  <summary><strong>1. VGG-16-BN/CIFAR-10</strong></summary>

<div align="center">

| Model                  | Top-1 (%) | # Params. (↓%) | FLOPs (↓%)  |
|------------------------|----------|--------------|------------|
| *VGG-16-BN*             | 93.96    | 14.98M(00.0) | 313.73M(00.0) |
| L1                      | 93.40    | 5.40M(64.0)  | 206.00M(34.3) |
| SSS                     | 93.02    | 3.93M(73.8)  | 183.13M(41.6) |
| GAL-0.05                | 92.03    | 3.36M(77.6)  | 189.49M(39.6) |
| VAS                     | 93.18    | 3.92M(73.3)  | 190.00M(39.1) |
| CHIP                    | 93.86    | 2.76M(81.6)  | 131.17M(58.1) |
| EZCrop                  | 93.01    | 2.76M(81.6)  | 131.17M(58.1) |
| DECORE-500              | 94.02    | 5.54M(63.0)  | 203.08M(35.3) |
| FPAC                    | 94.03    | 2.76M(81.6)  | 131.17M(58.1) |
| **CORING-C-5 (Ours)**   | **94.42**| **2.76M(81.6)**| **131.17M(58.1)**|
| GAL-0.1                 | 90.73    | 2.67M(82.2)  | 171.89M(45.2) |
| HRank-2                 | 92.34    | 2.64M(82.1)  | 108.61M(65.3) |
| HRank-1                 | 93.43    | 2.51M(82.9)  | 145.61M(53.5) |
| DECORE-200              | 93.56    | 1.66M(89.0)  | 110.51M(64.8) |
| EZCrop                  | 93.70    | 2.50M(83.3)  | 104.78M(66.6) |
| CHIP                    | 93.72    | 2.50M(83.3)  | 104.78M(66.6) |
| FSM                     | 93.73    | N/A(86.3)    | N/A(66.0)   |
| FPAC                    | 93.86    | 2.50M(83.3)  | 104.78M(66.6) |
| AutoBot                 | 94.01    | 6.44M(57.0)  | 108.71M(65.3) |
| **CORING-C-15 (Ours)**  | **94.20**| 2.50M(83.3)  | **104.78M(66.6)**|
| HRank-3                 | 91.23    | 1.78M(92.0)  | 73.70M(76.5)  |
| DECORE-50               | 91.68    | 0.26M(98.3)  | 36.85M(88.3)  |
| QSFM                    | 92.17    | 3.68M(75.0)  | 79.00M(74.8)  |
| DECORE-100              | 92.44    | 0.51M(96.6)  | 51.20M(81.5)  |
| FSM                     | 92.86    | N/A(90.6)    | N/A(81.0)    |
| CHIP                    | 93.18    | 1.90M(87.3)  | 66.95M(78.6)  |
| **CORING-C-10 (Ours)**  | **93.83**| 1.90M(87.3)  | 66.95M(78.6)  |
</div>

</details>

<details>
  <summary><strong>2. ResNet-56/CIFAR-10</strong></summary>

<div align="center">

| Model                     | Top-1(%) | # Params. (↓%) | FLOPs (↓%)      |
|---------------------------|---------|---------------|---------------|
| *ResNet-56*               | 93.26   | 0.85M(00.0)   | 125.49M(00.0) |
| L1                        | 93.06   | 0.73M(14.1)   | 90.90M(27.6)  |
| NISP                      | 93.01   | 0.49M(42.4)   | 81.00M(35.5)  |
| GAL-0.6                   | 92.98   | 0.75M(11.8)   | 78.30M(37.6)  |
| HRank-1                   | 93.52   | 0.71M(16.8)   | 88.72M(29.3)  |
| DECORE-450                | 93.34   | 0.64M(24.2)   | 92.48M(26.3)  |
| TPP                       | 93.81   | N/A           | N/A(31.1)     |
| **CORING-E-5 (Ours)**     | **94.76**| 0.66M(22.4)   | 91.23M(27.3)  |
| HRank-2                   | 93.17   | 0.49M(42.4)   | 62.72M(50.0)  |
| DECORE-200                | 93.26   | 0.43M(49.0)   | 62.93M(49.9)  |
| TPP                       | 93.46   | N/A           | N/A(49.8)     |
| FSM                       | 93.63   | N/A(43.6)     | N/A(51.2)     |
| CC-0.5                    | 93.64   | 0.44M(48.2)   | 60M(52.0)     |
| FPAC                      | 93.71   | 0.48M(42.8)   | 65.94M(47.4)  |
| ResRep                    | 93.71   | N/A           | 59.3M(52.7)   |
| DCP                       | 93.72   | N/A(49.7)     | N/A(54.8)     |
| EZCrop                    | 93.80   | 0.48M(42.8)   | 65.94M(47.4)  |
| CHIP                      | 94.16   | 0.48M(42.8)   | 65.94M(47.4)  |
| **CORING-V-5 (Ours)**     | **94.22**| 0.48M(42.8)   | 65.94M(47.4)  |
| GAL-0.8                   | 90.36   | 0.29M(65.9)   | 49.99M(60.2)  |
| HRank-3                   | 90.72   | 0.27M(68.1)   | 32.52M(74.1)  |
| DECORE-55                 | 90.85   | 0.13M(85.3)   | 23.22M(81.5)  |
| QSFM                      | 91.88   | 0.25M(71.3)   | 50.62M(60.0)  |
| CHIP                      | 92.05   | 0.24M(71.8)   | 34.79M(72.3)  |
| TPP                       | 92.35   | N/A           | N/A(70.6)     |
| FPAC                      | 92.37   | 0.24M(71.8)   | 34.79M(72.3)  |
| **CORING-E (Ours)**       | **92.84**| 0.24M(71.8)   | 34.79M(72.3)  |
</div>

</details>

<details>
  <summary><strong>3. DenseNet-40/CIFAR-10</strong></summary>

<div align="center">

| Model                                  | Top-1 (%) | # Params. (↓%) | FLOPs (↓%)       |
|----------------------------------------|----------|---------------|-----------------|
| *DenseNet-40*                          | 94.81    | 1.04M(00.0)   | 282.92M(00.0)  |
| DECORE-175                             | 94.85    | 0.83M(20.7)   | 228.96M(19.1)  |
| **CORING-C (Ours)**                    | **94.88**| **0.80M(23.1)**| **224.12M(20.8)** |
| GAL-0.01                               | 94.29    | 0.67M(35.6)   | 182.92M(35.3)  |
| HRank-1                                | 94.24    | 0.66M(36.5)   | 167.41M(40.8)  |
| FPAC                                   | 94.51    | 0.62M(40.1)   | 173.39M(38.5)  |
| DECORE-115                             | 94.59    | 0.56M(46.0)   | 171.36M(39.4)  |
| **CORING-E (Ours)**                    | **94.71**| 0.62M(40.4)   | 173.39M(38.8)  |
| FPAC                                   | 93.66    | 0.39M(61.9)   | 113.08M(59.9)  |
| HRank-2                                | 93.68    | 0.48M(53.8)   | 110.15M(61.0)  |
| EZCrop                                 | 93.76    | 0.39M(61.9)   | 113.08M(59.9)  |
| DECORE-70                              | 94.04    | 0.37M(65.0)   | 128.13M(54.7)  |
| **CORING-C (Ours)**                    | **94.20**| 0.45M(56.7)   | 133.17M(52.9)  |
</div>

</details>

<details>
  <summary><strong>4. GoogLeNet-40/CIFAR-10</strong></summary>

  <div align="center">

| Model                                   | Top-1 (%) | # Params. (↓%) | FLOPs (↓%)       |
|-----------------------------------------|----------|---------------|-----------------|
| *GoogLeNet*                             | 95.05    | 6.15M(00.0)   | 1.52B(00.0)     |
| DECORE-500                              | 95.20    | 4.73M(23.0)   | 1.22B(19.8)     |
| **CORING-V (Ours)**                     | **95.30**| **4.72M(23.3)**| **1.21B(20.4)** |
| L1                                      | 94.54    | 3.51M(42.9)   | 1.02B(32.9)     |
| GAL-0.05                                | 93.93    | 3.12M(49.3)   | 0.94B(38.2)     |
| HRank-1                                 | 94.53    | 2.74M(55.4)   | 0.69M(54.9)     |
| FPAC                                    | 95.04    | 2.85M(53.5)   | 0.65B(57.2)     |
| CC-0.5                                  | 95.18    | 2.83M(54.0)   | 0.76B(50.0)     |
| **CORING-E (Ours)**                     | **95.32**| **2.85M(53.5)**| **0.65B(57.2)** |
| HRank-2                                 | 94.07    | 1.86M(69.8)   | 0.45B(70.4)     |
| FSM                                     | 94.29    | N/A(64.6)     | N/A(75.4)       |
| DECORE-175                              | 94.33    | 0.86M(86.1)   | 0.23B(84.7)     |
| FPAC                                    | 94.42    | 2.09M(65.8)   | 0.40B(73.9)     |
| DECORE-200                              | 94.51    | 1.17M(80.9)   | 0.33B(78.5)     |
| CLR-RNF-0.91                            | 94.85    | 2.18M(64.7)   | 0.49B(67.9)     |
| CC-0.6                                  | 94.88    | 2.26M(63.3)   | 0.61B(59.9)     |
| **CORING-E (Ours)**                     | **95.03**| 2.10M(65.9)   | 0.39B(74.3)     |
</div>

</details>

<details>
  <summary><strong>5. MobileNetv2/CIFAR-10</strong></summary>

<div align="center">

| Model                | Top-1 (%) | # Params (↓%) | FLOPs (↓%)      |
|----------------------|----------|--------------|----------------|
| *MobileNetv2*        | 94.43    | 2.24M(0.0)   | 94.54M(0.0)    |
| DCP                  | 94.02    | N/A(23.6)    | N/A(26.4)      |
| WM                   | 94.02    | N/A          | N/A(27.0)      |
| QSFM-PSNR            | 92.06    | 1.67M(25.4)  | 57.27M(39.4)   |
| DMC                  | 94.49    | N/A          | N/A(40.0)      |
| SCOP                 | 94.24    | N/A(36.1)    | N/A(40.3)      |
| GFBS                 | 94.25    | N/A          | N/A(42.0)      |
| **CORING-V (Ours)**  | **94.81**| **1.26M(43.8)**| **55.16M(42.0)**|
| **CORING-V (Ours)**  | **94.44**| **0.77M(65.6)**| **38.00M(60.0)**|
</div>

</details>

<details>
  <summary><strong>6. Resnet-50/Imagenet</strong></summary>

<div align="center">

| Model                               | Top-1 (%) | Top-5 (%) | # Params (↓%)      | FLOPs (↓%)         |
|-------------------------------------|----------|----------|-------------------|-------------------|
| *ResNet-50*                         | 76.15    | 92.87    | 25.50M(0.0)       | 4.09B(0.0)        |
| AutoPruner-0.3                      | 74.76    | 92.15    | N/A               | 3.76B(8.1)        |
| ABCPruner-100%                      | 72.84    | 92.97    | 18.02M(29.3)      | 2.56B(37.4)       |
| CLR-RNF-0.2                         | 74.85    | 92.31    | 16.92M(33.6)      | 2.45B(40.1)       |
| APRS                                | 75.58    | N/A      | 16.17M(35.4)      | 2.29B(44.0)       |
| PFP                                 | 75.91    | 92.81    | 20.88M(18.1)      | 3.65B(10.8)       |
| LeGR                                | 76.20    | 93.00    | N/A               | N/A(27.0)         |
| DECORE-8                            | 76.31    | 93.02    | 22.69M(11.0)      | 3.54B(13.4)       |
| CHIP                                | 76.30    | 93.02    | 15.10M(40.8)      | 2.26B(44.8)       |
| TPP                                 | 76.44    | N/A      | N/A               | N/A(32.9)         |
| **CORING-V (Ours)**                 | **76.78**| **93.23**| **15.10M(40.8)** | **2.26B(44.8)**   |
| GAL-0.5                             | 71.95    | 90.94    | 21.20M(16.9)      | 2.33B(43.0)       |
| AutoPruner-0.5                      | 73.05    | 91.25    | N/A               | 2.64B(35.5)       |
| HRank-1                             | 74.98    | 92.33    | 16.15M(36.7)      | 2.30B(43.8)       |
| DECORE-6                            | 74.58    | 92.18    | 14.10M(44.7)      | 2.36B(42.3)       |
| PFP                                 | 75.21    | 92.43    | 17.82M(30.1)      | 2.29B(44.0)       |
| FPAC                                | 75.62    | 92.63    | 15.09M(40.9)      | 2.26B(45.0)       |
| EZCrop                              | 75.68    | 92.70    | 15.09M(40.9)      | 2.26B(45.0)       |
| LeGR                                | 75.70    | 92.70    | N/A               | N/A(42.0)         |
| SCOP                                | 75.95    | 92.79    | 14.59M(42.8)      | 2.24B(45.3)       |
| CHIP                                | 76.15    | 92.91    | 14.23M(44.2)      | 2.10B(48.7)       |
| **CORING-C (Ours)**                 | **76.34**| **93.06**| **14.23M(44.2)** | **2.10B(48.7)**   |
| GAL-0.5-joint                       | 71.80    | 89.12    | 19.31M(24.3)      | 1.84B(55.0)       |
| HRank-2                             | 71.98    | 91.01    | 13.77M(46.0)      | 1.55B(62.1)       |
| MFMI                                | 72.02    | 90.69    | 11.41M(55.2)      | 1.84B(55.0)       |
| FPAC                                | 74.17    | 91.84    | 11.05M(56.7)      | 1.52B(62.8)       |
| EZCrop                              | 74.33    | 92.00    | 11.05M(56.7)      | 1.52B(62.8)       |
| CC-0.6                              | 74.54    | 92.25    | 10.58M(58.5)      | 1.53B(62.6)       |
| APRS                                | 74.72    | N/A      | N/A               | N/A(57.2)         |
| TPP                                 | 75.12    | N/A      | N/A               | N/A(60.9)         |
| SCOP                                | 75.26    | 92.53    | 12.29M(51.8)      | 1.86B(54.6)       |
| CHIP                                | 75.26    | 92.53    | 11.04M(56.7)      | 1.52B(62.8)       |
| LeGR                                | 75.30    | 92.40    | N/A               | N/A(53.0)         |
| ResRep                              | 75.30    | 92.47    | N/A               | 1.52B(62.1)       |
| **CORING-V (Ours)**                 | **75.55**| **92.61**| **11.04M(56.7)** | **1.52B(62.8)**   |
| GAL-1-joint                         | 69.31    | 89.12    | 10.21M(60.0)      | 1.11B(72.9)       |
| HRank-3                             | 69.10    | 89.58    | 8.27M(67.6)       | 0.98B(76.0)       |
| DECORE-4                            | 69.71    | 89.37    | 6.12M(76.0)       | 1.19B(70.9)       |
| MFMI                                | 69.91    | 89.46    | 8.51M(66.6)       | 1.41B(34.4)       |
| DECORE-5                            | 72.06    | 90.82    | 8.87M(65.2)       | 1.60B(60.9)       |
| FPAC                                | 72.30    | 90.74    | 8.02M(68.6)       | 0.95B(76.7)       |
| ABCPruner-50%                       | 72.58    | 90.91    | 9.10M(64.3)       | 1.30B(68.2)       |
| CHIP                                | 72.30    | 90.74    | 8.01M(68.6)       | 0.95B(76.7)       |
| CLR-RNF-0.44                        | 72.67    | 91.09    | 9.00M(64.7)       | 1.23B(69.9)       |
| CURL                                | 73.39    | 91.46    | 6.67M(73.8)       | 1.11B(72.9)       |
| **CORING-V (Ours)**                 | **73.99**| **91.71**| **8.01M(68.6)**  | **0.95B(76.7)**   |
</div>

</details>


# Installation
Main requirements:
```
- python=3.9
- pytorch >= 1.13
- tensorly=0.7
- numpy=1.21
- thop=0.1
- ptflops
```


# Verification of our results
  Please download the [checkpoints](https://drive.google.com/drive/folders/1UfDCJ2x-Tp-4m51AfuTie_gV2N-YNYTl?usp=sharing) and evaluate their performance with the corresponding script and dataset.

- All results are available [here](https://drive.google.com/drive/folders/1UfDCJ2x-Tp-4m51AfuTie_gV2N-YNYTl?usp=sharing).

  <details>

  *Notes:* Log files of all experiments are attached, they contain all information about the pruning or fine-tuning process, as well as model architecture, numbers of parameters/FLOPs, and top-1/top-5 accuracy.
  </details>

- Download the datasets
  <details>

   The CIFAR dataset will be automatically downloaded.

   The Imagenet dataset can be downloaded [here](https://image-net.org/download-images.php) and processed as this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
  </details>

- Use [main/test.py](main/test.py) to validate the performance of the checkpoints.

  <details>

  ```bash
    python main/test.py --dataset cifar10 --data_dir data/cifar10 --arch vgg_16_bn --compress_rate [0.21]*7+[0.75]*5 --model_path ./pruned_model/cifar10/vgg16bn/soft/model_best.pth.tar
    python main/test.py --dataset cifar10 --data_dir data/cifar10 --arch resnet_56 --compress_rate [0.]+[0.18]*29 --model_path ./pruned_model/cifar10/resnet56/soft/model_best.pth.tar
    python main/test.py --dataset cifar10 --data_dir data/cifar10 --arch densenet_40 --compress_rate [0.]+[0.08]*6+[0.09]*6+[0.08]*26 --model_path ./pruned_model/cifar10/densenet/soft/model_best.pth.tar
    python main/test.py --dataset cifar10 --data_dir data/cifar10 --arch mobilenet_v2 --compress_rate [0.]+[0.1]+[0.25]*2+[0.25]*2+[0.3]*2 --model_path ./pruned_model/cifar10/mobilenetv2/moderate/model_best.pth.tar
    python main/test.py --dataset imagenet --data_dir data/imagenet --arch resnet_50 --compress_rate [0.]+[0.5]*3+[0.6]*16 --model_path ./pruned_model/imagenet/extreme/model_best.pth.tar
  ```
  </details>


# Reproducibility and further development
1. To reproduce results, you may run prepared scripts.
- For CIFAR-10, the rank will be calculated during the pruning process.
- For Resnet50/ImageNet, first, generate the rank by this:
    ```
    sh main/scripts/generate_rank_resnet50.sh
    ```
- Now, the pruning process can be performed via prepared scripts. For example:
    ```
    sh main/scripts/resnet50_imagenet/vbd.sh
    ```
2. Our code is pipelined and can be integrated into other works. Just replace the filter ranking computation.
    ```
    # replace your rank calculation here
    rank = get_rank(oriweight, args.criterion, args.strategy)
    ```


# :art: Supplementary materials
1. **Poster**
<div>
  <img class="image" src="assets\poster.png" width="100%" height="100%">
</div>

2. **Architecture constraint**
<p align="center" width="100%">
    <img src="assets\residual.png" width="60%" height="60%">
</p>

With shortcut connection architecture, input and output of each residual block are forced identical. In each layer (*i.e,* same color), filters with same style (*e.g,* sketch) are highly similar, and the empty dashed one are to be pruned. After pruning, the input and output layer (*green and red*) has the same number of filters.

3. **Computational requirement comparison**
<p align="center" width="100%">
    <img src="assets\time_benchmark.png" width="50%" height="50%">
</p>

Time consumption to calculate the similarity matrix on VGG-16-BN. For tail layers that contain a larger number of filters, the tensor decomposition method is obviously more efficient.

4. **Criteria comparison.**
<p align="center" width="100%">
    <img src="assets\criteria_comparison.png" width="50%" height="50%">
</p>

A comprehensive ablation study on CIFAR-10, showcasing comparable final accuracies achieved with the 3 considered distances.


# :bookmark_tabs: ToDo
- [ ] Integrate other pruning techniques.
- [ ] Clean code.

# :email: Contact
 We hope that the new perspective of CORING and its template may inspire more developments :rocket: on network compression.

We warmly welcome your participation in our project!

To contact us, never hesitate to contact [pvtien96@gmail.com](mailto:pvtien96@gmail.com).
<br></br>

# Citation
If the code and paper help your research, please kindly cite:
```
@misc{pham2023coring,
    title={Efficient tensor decomposition-based filter pruning}, 
    author={Van Tien, Pham and Yassine, Zniyed and Thanh Phuong, Nguyen},
    year={2023},
    howpublished={\url{https://github.com/pvtien96/CORING}},
  }
```

# Acknowledgement
Part of this repository is based on [HRankPlus](https://github.com/lmbxmu/HRankPlus/tree/master).
