<div align="center">

# (NeurIPS 2025) Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13667-b31b1b.svg)](https://arxiv.org/abs/2505.24003)
[![PyPI - Version](https://img.shields.io/pypi/v/version)](#package)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue)]()
</div>

<div align="center">

**[<a href="https://mp.weixin.qq.com/s/Yu1uE-xxGTbER7oQVtX2Fw">时序人</a>]**
**[<a href="https://mp.weixin.qq.com/s/gld2cZam7GyAY3PlEh7P0w">时空探索之旅</a>]**
**[<a href="https://mp.weixin.qq.com/s/mB2UtxhPWnFomIkr5Ozf2A">QuantML</a>]**
**[<a href="https://mp.weixin.qq.com/s/pRGQxicQnBEEAUCHEuSeVA">时序之心</a>]**
**[<a href="https://mp.weixin.qq.com/s/sk8K_M7nKCYdZNhQ1EGHMQ">时序大模型</a>]**

</div>

<p align="center">
<p align="center">
    🔍&nbsp;<a href="#about">About</a>
    | 🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 📊&nbsp;<a href="#results-and-evaluation">Evaluation</a>
    | 🔗&nbsp;<a href="#citation">Citation</a>
</p>

## 🔍About

This is the official repository for NeurIPS 2025 paper "[Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting](https://arxiv.org/abs/2505.24003)". This paper proposes DMMV, a novel decomposition-based multi-modal
view (MMV) framework that leverages trend-seasonal decomposition and a novel backcast-residual based adaptive decomposition to integrate MMVs of time series and large vision models (LVMs) for long-term time series forecasting (LTSF).

### 🔧Framework

Traditional time series forecasting models often rely on a single view (e.g., numerical, language, visual), overlooking the complementary information that can be integrated across different modalities. The proposed Decomposition-based Multi-Modal View (DMMV) framework addresses this limitation by jointly modeling the numerical and visual views of time series within a unified architecture.

As illustrated in Figure 1, DMMV consists of two variants, DMMV-S and DMMV-A.
- **DMMV-S**: Uses a moving-average kernel to decompose the series into seasonal and trend parts, processed by the Visual and Numerical Forecasters, respectively.
- **DMMV-A**: Leverages the Visual Forecaster for both forecasting and backcasting to reconstruct seasonal components, while adaptively using the Numerical Forecaster to model the residual trend component.

Both variants share the following core components:
- **Visual Forecaster**: Utilizes a pre-trained LVM to reconstruct the masked regions of input imaged time series, effectively capturing periodic and local patterns.
- **Numerical Forecaster**: A general series-to-series predictor that models global trends. It can be implemented as a linear-layer or Transformer-based forecaster that reads the numerical view of time series.
- **Fusion Gate**: An adaptive gating mechanism that integrates the outputs from both forecasters, balancing trend and periodic information to produce the final forecast.


<div align="center">

|[<img src="./image/model_framework.png" width=90%/>](./image/model_framework.png)|
|:--:|
|Figure 1: An overview of DMMV framework. (a) DMMV-S uses moving-average to extract trend and seasonal components. (b) DMMV-A uses a backcast-residual decomposition to automatically learn trend and seasonal components. In (b), the gray blocks are gray-scale images. "?" marks masks.|
</div>


### 🔑 Key Features

- **Multi-Modal Integration**: Jointly models numerical and visual views of time series while making use of the strengths of LVM forecasters and numerical forecasters.
- **Decomposition**: Introduces a novel adaptive backcast–residual decomposition framework that can harness LVMs’ inductive biases.
- **Modular Compatibility**: Supports various LVMs (e.g., MAE, SimMIM) and numerical forecasters (e.g., Linear, PatchTST) for flexible deployment.

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/D2I-Group/dmmv.git
   cd dmmv
   ```

2. **Set Up the Environment**
   - We recommend using `conda` or `virtualenv` to create an isolated environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Download the datasets**
    - You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322) provided by [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
    - Then place the downloaded data in the folder `./dataset`. 
    - Here is a summary of the benchmark datasets.

    <p align="center">
        <img src="./image/table1.png" alt="table1" width="90%">
    </p>

4. **Run the Code**
    - Make sure the environment and settings are correctly configured.
    - Run `bash scripts/DMMV-A/ETTh1.sh` to start.



## 📊 Evaluation

DMMV is comprehensively compared with 14 state-of-the-art (SOTA) models on 8 benchmark datasets across domains. The baseline methods cover different time series forecasting models, including LLM-, LVM-, VLM-, Transformer-, CNN-, and MLP-based methods. DMMV achieves the best mean squared error (MSE) on 6 out of 8 datasets.
Figure 2 presents the ranking of DMMV and the baseline methods in terms of MSE and mean absolute error (MAE), providing an overview of DMMV's performance.


<div align="center">

|[<img src="./image/rank.png" width=90%/>](./image/rank.png)|
|:--:|
|Figure 2: Critical difference (CD) diagram on the average rank of all 16 compared methods in terms of (a) MSE and (b) MAE over all benchmark datasets. The lower rank (left of the scale) is better.|
</div>

<p align="center">
    <img src="./image/table2.png" alt="table2" width="90%">
</p>


## 🔗 Citation

```
@inproceedings{shen2025dmmv,
      title={Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting}, 
      author={ChengAo Shen and Wenchao Yu and Ziming Zhao and Dongjin Song and Wei Cheng and Haifeng Chen and Jingchao Ni},
      booktitle={NeurIPS},
      year={2025},
}
```
## 📧 Contact

If you have any questions or concerns, please contact us: cshen9 [at] uh [dot] edu or submit an issue
