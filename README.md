# Semi-supervised-Learning-for-Acoustic-Impedance-Inversion
[Motaz Alfarraj](http://www.motaz.me), and [Ghassan AlRegib](http://www.ghassanalregib.info)

Codes and data for a manuscript published in SEG Annual Meeting 2019, San Antonio, TX

This repository contains the codes for the paper: 

M. Alfarraj and G. AlRegib, "**Semi-Supervised Learning for Acoustic Impedance Inversion**," in *Expanded Abstracts of the SEG Annual Meeting*, Sep. 15-20 2019. [[SEG Digital Library]](https://library.seg.org/doi/10.1190/segam2019-3215902.1) [[ArXiv]](https://arxiv.org/pdf/1905.13412)

## Abstract
Recent applications of deep learning in the seismic domain have shown great potential in different areas such as inversion and interpretation. Deep learning algorithms, in general, require tremendous amounts of labeled data to train properly. To overcome this issue, we propose a semi-supervised framework for acoustic impedance inversion based on convolutional and recurrent neural networks. Specifically, seismic traces and acoustic impedance traces are modeled as time series. Then, a neural-network-based inversion model comprising convolutional and recurrent neural layers is used to invert seismic data for acoustic impedance. The proposed workflow uses well log data to guide the inversion. In addition, it utilizes a learned seismic forward model to regularize the training and to serve as a geophysical constraint for the inversion. The proposed workflow achieves an average correlation of 98% between the estimated and target elastic impedance using 20 AI traces for training. 


## Data 
The data used in this code are from the elastic model of [Marmousi 2](https://library.seg.org/doi/abs/10.1190/1.1817083)
The data file can be found under /data/ 

Both acoustic impedance and seismic are saved in the same `data.npy` file.

## Running the code

### Requirements: 
These are the python libraries that are needed to run the code. Newer version should work fine as well. 
```
matplotlib==3.1.1
numpy==1.17.0
pyparsing==2.4.1.1
python-dateutil==2.8.0
torch==1.1.0
torchvision==0.3.0
tqdm==4.33.0
wget==3.2
```
Note: This code is built using [PyTorch](https://pytorch.org/) with GPU support. Follow the instructions on PyTorch's website to install it properly. The code can also be run without GPU, but it will be much slower. 

### Training and testing

To train the model using the default parameters (as reported in the paper), and test it on the full Marmousi 2 model, run the following command: 

```bash 
python main.py
```
 However, you can choose those parameters by including the arguments and their values. For example, to change the number of training traces, you can run: 
 
```bash 
python main.py -num_train_wells 20
```

The list arguments can be found in the file `main.py`.  



## Citation: 

If you have found our code and data useful, we kindly ask you to cite our work 
```tex
@inbook{doi:10.1190/segam2019-3215902.1,
author = {Motaz Alfarraj and Ghassan AlRegib},
title = {Semi-supervised learning for acoustic impedance inversion},
booktitle = {SEG Technical Program Expanded Abstracts 2019},
chapter = {},
pages = {2298-2302},
year = {2019},
doi = {10.1190/segam2019-3215902.1},
URL = {https://library.seg.org/doi/abs/10.1190/segam2019-3215902.1},
eprint = {https://library.seg.org/doi/pdf/10.1190/segam2019-3215902.1},
    abstract = { Recent applications of deep learning in the seismic domain have shown great potential in different areas such as inversion and interpretation. Deep learning algorithms, in general, require tremendous amounts of labeled data to train properly. To overcome this issue, we propose a semi-supervised framework for acoustic impedance inversion based on convolutional and recurrent neural networks. Specifically, seismic traces and acoustic impedance traces are modeled as time series. Then, a neural-network-based inversion model comprising convolutional and recurrent neural layers is used to invert seismic data for acoustic impedance. The proposed workflow uses well log data to guide the inversion. In addition, it utilizes a learned seismic forward model to regularize the training and to serve as a geophysical constraint for the inversion. The proposed workflow achieves an average correlation of 98\% between the estimated and target elastic impedance using 20 AI traces for training.Presentation Date: Tuesday, September 17, 2019Session Start Time: 8:30 AMPresentation Time: 9:45 AMLocation: 221DPresentation Type: Oral }
}

```
