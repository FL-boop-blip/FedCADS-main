# FedCADS: Robust Federated Learning via Dual Distillation and Participation-Aware Optimization under Non-IID Data
Code for paper - **[FedCADS: Robust Federated Learning via Dual Distillation and Participation-Aware Optimization under Non-IID Data]**

We provide code to run FedCADS,
[A_FedPD](https://openreview.net/pdf?id=h1iMVi2iEM),
[FedGKD + PD](https://www.researchgate.net/profile/Dezhong-Yao/publication/373957735_FedGKD_Towards_Heterogeneous_Federated_Learning_via_Global_Knowledge_Distillation/links/65fae109a4857c7962653369/FedGKD-Toward-Heterogeneous-Federated-Learning-via-Global-Knowledge-Distillation.pdf),
[FedDisco](https://proceedings.mlr.press/v202/ye23f/ye23f.pdf),
[FedVRA](https://ojs.aaai.org/index.php/AAAI/article/view/26212),
[FedDC](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf), 
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf), 
[FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), 
[Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), and [FedProx](https://arxiv.org/abs/1812.06127) methods.


## Prerequisite
* Install the libraries listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```

## Datasets preparation
**We give datasets for the benchmark, including CIFAR10 and CIFAR100 dataset.**




You can obtain the datasets when you first time run the code on CIFAR10, CIFAR100 datasets.


For example, you can follow the following steps to run the experiments:

```python example_code_cifar10.py```
```python example_code_cifar100.py```


1. Run the following script to run experiments on CIFAR10 for all above methods:
    ```
    python example_code_cifar10.py
    ```
2. Run the following script to run experiments on CIFAR100 for all above methods:
    ```
    python example_code_cifar100.py
    ```
3. To show the convergence plots, we use the tensorboardX package. As an example to show the results which stored in "./Folder/Runs/CIFAR100_100_20_iid_":
    ```
    tensorboard --logdir=./Folder/Runs/CIFAR100_100_20_iid_
    ```
4Get the url, and then enter the url in to the web browser, for example "http://localhost:6006/".

   
## Generate IID, Dirichlet and Pathological distributions:
Modify the DatasetObject() function in the example code.
CIFAR-10 IID, 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=20, rule='iid', unbalanced_sgm=0, data_path=data_path)
```
CIFAR-10 Dirichlet (0.3), 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
```

    
## FedCADS 
The FedCADS method is implemented in ```utils_methods_FedCADS.py```. The baseline methods are stored in ```utils_methods.py```.

## Some Experiments
The accuracy and training time with different methods using LeNet. The training time is obtained on RTX4090.
<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">   </td>
            <td colspan="10"> CIFAR-10 (LeNet) </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="5">  10%-100 (bs=50 Local-epoch=5 T=1000)  </td>
            <td colspan="5">  2%-500 (bs=50 Local-epoch=5 T=1600)	 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> P-6 </td>
            <td colspan="1"> P-3 </td>
            <td colspan="1"> Time / round </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> P-6 </td>
            <td colspan="1"> P-3 </td>
            <td colspan="1"> Time / round </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> 80.95 </td>
            <td colspan="1"> 79.64 </td>
            <td colspan="1"> 81.34 </td>
            <td colspan="1"> 78.63 </td>
            <td colspan="1"> 3.45s </td>
            <td colspan="1"> 73.84 </td>
            <td colspan="1"> 72.75 </td>
            <td colspan="1"> 74.78 </td>
            <td colspan="1"> 72.05 </td>
            <td colspan="1"> 3.74s </td>
        </tr>
        <tr>
            <td colspan="1"> FedProx </td>
            <td colspan="1"> 80.22 </td>
            <td colspan="1"> 78.22 </td>
            <td colspan="1"> 79.55 </td>
            <td colspan="1"> 74.47 </td>
            <td colspan="1"> 3.61s </td>
            <td colspan="1"> 73.61 </td>
            <td colspan="1"> 72.82 </td>
            <td colspan="1"> 74.49 </td>
            <td colspan="1"> 71.73 </td>
            <td colspan="1"> 3.84s </td>
        </tr>
        <tr>
            <td colspan="1"> SCAFFOLD </td>
            <td colspan="1"> 83.36 </td>
            <td colspan="1"> 82.32 </td>
            <td colspan="1"> 83.70 </td>
            <td colspan="1"> 80.69 </td>
            <td colspan="1"> 4.51s </td>
            <td colspan="1"> 78.50 </td>
            <td colspan="1"> 77.24 </td>
            <td colspan="1"> 78.64 </td>
            <td colspan="1"> 75.05 </td>
            <td colspan="1"> 4.64s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> 83.30 </td>
            <td colspan="1"> 82.06 </td>
            <td colspan="1"> 83.92 </td>
            <td colspan="1"> 81.32 </td>
            <td colspan="1"> 4.38s </td>
            <td colspan="1"> 74.76 </td>
            <td colspan="1"> 72.71 </td>
            <td colspan="1"> 73.95 </td>
            <td colspan="1"> 66.44 </td>
            <td colspan="1"> 4.48s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDC </td>
            <td colspan="1"> 84.03 </td>
            <td colspan="1"> 83.22 </td>
            <td colspan="1"> 84.10 </td>
            <td colspan="1"> 82.24 </td>
            <td colspan="1"> 5.96s </td>
            <td colspan="1"> 77.36 </td>
            <td colspan="1"> 75.78 </td>
            <td colspan="1"> 77.77 </td>
            <td colspan="1"> 75.23 </td>
            <td colspan="1"> 6.58s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDisco </td>
            <td colspan="1"> 82.09 </td>
            <td colspan="1"> 81.09 </td>
            <td colspan="1"> 81.78 </td>
            <td colspan="1"> 76.82 </td>
            <td colspan="1"> 6.73s </td>
            <td colspan="1"> 74.84 </td>
            <td colspan="1"> 72.73 </td>
            <td colspan="1"> 74.46 </td>
            <td colspan="1"> 66.60 </td>
            <td colspan="1"> 7.06s </td>
        </tr>
        <tr>
            <td colspan="1"> FedVRA </td>
            <td colspan="1"> 83.12 </td>
            <td colspan="1"> 81.86 </td>
            <td colspan="1"> 82.77 </td>
            <td colspan="1"> 73.75 </td>
            <td colspan="1"> 4.40s </td>
            <td colspan="1"> 74.65 </td>
            <td colspan="1"> 72.34 </td>
            <td colspan="1"> 74.20 </td>
            <td colspan="1"> 65.74 </td>
            <td colspan="1"> 4.45s </td>
        </tr>
        <tr>
            <td colspan="1"> FedGKD_PD </td>
            <td colspan="1"> 83.39 </td>
            <td colspan="1"> 82.31 </td>
            <td colspan="1"> 84.02 </td>
            <td colspan="1"> 82.12 </td>
            <td colspan="1"> 5.58s </td>
            <td colspan="1"> 75.27 </td>
            <td colspan="1"> 74.23 </td>
            <td colspan="1"> 74.92 </td>
            <td colspan="1"> 69.38 </td>
            <td colspan="1"> 6.04s </td>
        </tr>
        <tr>
            <td colspan="1"> A_FedPD </td>
            <td colspan="1"> 83.36 </td>
            <td colspan="1"> 81.83 </td>
            <td colspan="1"> 83.40 </td>
            <td colspan="1"> 79.44 </td>
            <td colspan="1"> 5.47s </td>
            <td colspan="1"> 76.32 </td>
            <td colspan="1"> 75.27 </td>
            <td colspan="1"> 76.24 </td>
            <td colspan="1"> 72.69 </td>
            <td colspan="1"> 6.02s </td>
        </tr>
        <tr>
            <td colspan="1"> FedCADS </td>
            <td colspan="1"> 84.02 </td>
            <td colspan="1"> 82.73 </td>
            <td colspan="1"> 84.53 </td>
            <td colspan="1"> 82.19 </td>
            <td colspan="1"> 5.97s </td>
            <td colspan="1"> 82.44 </td>
            <td colspan="1"> 81.06 </td>
            <td colspan="1"> 82.36 </td>
            <td colspan="1"> 79.42 </td>
            <td colspan="1"> 6.48s </td>
        </tr>
    </tbody>
</table>
</p>

The accuracy and training time with different methods using ResNet18. The training time is obtained on RTX4090.

<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">   </td>
            <td colspan="10"> CIFAR-10 (ResNet18) </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="5">  10%-100 (bs=50 Local-epoch=5 T=1000)  </td>
            <td colspan="5">  2%-500 (bs=50 Local-epoch=5 T=1600)	 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> P-6 </td>
            <td colspan="1"> P-3 </td>
            <td colspan="1"> Time / round </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> P-6 </td>
            <td colspan="1"> P-3 </td>
            <td colspan="1"> Time / round </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> 80.73 </td>
            <td colspan="1"> 79.27 </td>
            <td colspan="1"> 80.92 </td>
            <td colspan="1"> 73.86 </td>
            <td colspan="1"> 5.90s </td>
            <td colspan="1"> 72.30 </td>
            <td colspan="1"> 74.65 </td>
            <td colspan="1"> 72.75 </td>
            <td colspan="1"> 69.45 </td>
            <td colspan="1"> 6.12s </td>
        </tr>
        <tr>
            <td colspan="1"> FedProx </td>
            <td colspan="1"> 80.26 </td>
            <td colspan="1"> 78.54 </td>
            <td colspan="1"> 80.10 </td>
            <td colspan="1"> 72.74 </td>
            <td colspan="1"> 6.17s </td>
            <td colspan="1"> 75.20 </td>
            <td colspan="1"> 74.80 </td>
            <td colspan="1"> 73.00 </td>
            <td colspan="1"> 49.44 </td>
            <td colspan="1"> 6.34s </td>
        </tr>
        <tr>
            <td colspan="1"> SCAFFOLD </td>
            <td colspan="1"> 83.23 </td>
            <td colspan="1"> 81.62 </td>
            <td colspan="1"> 82.96 </td>
            <td colspan="1"> 74.67 </td>
            <td colspan="1"> 7.72s </td>
            <td colspan="1"> 76.55 </td>
            <td colspan="1"> 78.22 </td>
            <td colspan="1"> 77.46 </td>
            <td colspan="1"> 71.27 </td>
            <td colspan="1"> 7.94s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> 82.79 </td>
            <td colspan="1"> 82.01 </td>
            <td colspan="1"> 82.72 </td>
            <td colspan="1"> 78.02 </td>
            <td colspan="1"> 7.39s </td>
            <td colspan="1"> 76.48 </td>
            <td colspan="1"> 73.37 </td>
            <td colspan="1"> 74.00 </td>
            <td colspan="1"> 69.42 </td>
            <td colspan="1"> 7.56s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDC </td>
            <td colspan="1"> 84.44 </td>
            <td colspan="1"> 83.39 </td>
            <td colspan="1"> 84.97 </td>
            <td colspan="1"> 79.90 </td>
            <td colspan="1"> 9.78s </td>
            <td colspan="1"> 79.98 </td>
            <td colspan="1"> 79.06 </td>
            <td colspan="1"> 76.90 </td>
            <td colspan="1"> 73.40 </td>
            <td colspan="1"> 10.25s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDisco </td>
            <td colspan="1"> 83.09 </td>
            <td colspan="1"> 82.73 </td>
            <td colspan="1"> 82.56 </td>
            <td colspan="1"> 76.50 </td>
            <td colspan="1"> 11.26s </td>
            <td colspan="1"> 77.32 </td>
            <td colspan="1"> 73.45 </td>
            <td colspan="1"> 74.35 </td>
            <td colspan="1"> 68.93 </td>
            <td colspan="1"> 12.36s </td>
        </tr>
        <tr>
            <td colspan="1"> FedVRA </td>
            <td colspan="1"> 83.05 </td>
            <td colspan="1"> 82.26 </td>
            <td colspan="1"> 82.79 </td>
            <td colspan="1"> 78.69 </td>
            <td colspan="1"> 7.74s </td>
            <td colspan="1"> 77.34 </td>
            <td colspan="1"> 74.72 </td>
            <td colspan="1"> 74.68 </td>
            <td colspan="1"> 70.28 </td>
            <td colspan="1"> 7.46s </td>
        </tr>
        <tr>
            <td colspan="1"> FedGKD_PD </td>
            <td colspan="1"> 82.98 </td>
            <td colspan="1"> 82.31 </td>
            <td colspan="1"> 82.82 </td>
            <td colspan="1"> 78.83 </td>
            <td colspan="1"> 8.42s </td>
            <td colspan="1"> 77.53 </td>
            <td colspan="1"> 74.07 </td>
            <td colspan="1"> 74.69 </td>
            <td colspan="1"> 70.34 </td>
            <td colspan="1"> 9.76s </td>
        </tr>
        <tr>
            <td colspan="1"> A_FedPD </td>
            <td colspan="1"> 84.88 </td>
            <td colspan="1"> 83.73 </td>
            <td colspan="1"> 85.23 </td>
            <td colspan="1"> 79.01 </td>
            <td colspan="1"> 8.20s </td>
            <td colspan="1"> 75.14 </td>
            <td colspan="1"> 74.93 </td>
            <td colspan="1"> 75.52 </td>
            <td colspan="1"> 74.62 </td>
            <td colspan="1"> 11.75s </td>
        </tr>
        <tr>
            <td colspan="1"> FedCADS </td>
            <td colspan="1"> 85.05 </td>
            <td colspan="1"> 83.83 </td>
            <td colspan="1"> 85.11 </td>
            <td colspan="1"> 82.31 </td>
            <td colspan="1"> 9.31s </td>
            <td colspan="1"> 82.26 </td>
            <td colspan="1"> 81.77 </td>
            <td colspan="1"> 82.53 </td>
            <td colspan="1"> 81.03 </td>
            <td colspan="1"> 9.77s </td>
        </tr>
    </tbody>
</table>
</p>

Here is the accuracy of FedCADS under 100 clients with various client participation rate.

<p align="center">
<table>
    <tbody align="center" valign="center">
    </tr>
        <td colspan="1">   </td>
        <td colspan="6"> CIFAR-10 (ResNet18 Dir-0.6)  </td>
    </tr>
        <td colspan="1">  </td>
        <td colspan="1"> 5% </td>
        <td colspan="1"> 10% </td>
        <td colspan="1"> 20% </td>
        <td colspan="1"> 50% </td>
        <td colspan="1"> 80% </td>
        <td colspan="1"> 100% </td>
    </tr>
    </tr>
        <td colspan="1"> FedAvg </td>
        <td colspan="1"> 80.26 </td>
        <td colspan="1"> 80.73 </td>
        <td colspan="1"> 81.42 </td>
        <td colspan="1"> 80.82 </td>
        <td colspan="1"> 81.16 </td>
        <td colspan="1"> 81.02 </td>
    </tr>
    </tr>
        <td colspan="1"> FedProx </td>
        <td colspan="1"> 80.11 </td>
        <td colspan="1"> 80.26 </td>
        <td colspan="1"> 81.01 </td>
        <td colspan="1"> 81.12 </td>
        <td colspan="1"> 81.25 </td>
        <td colspan="1"> 81.22 </td>
    </tr>
    </tr>
        <td colspan="1"> SCAFFOLD </td>
        <td colspan="1"> 82.96 </td>
        <td colspan="1"> 83.23 </td>
        <td colspan="1"> 83.83</td>
        <td colspan="1"> 84.34 </td>
        <td colspan="1"> 84.15 </td>
        <td colspan="1"> 84.15 </td>
    </tr>
    </tr>
        <td colspan="1"> FedDyn </td>
        <td colspan="1"> 81.12 </td>
        <td colspan="1"> 82.79 </td>
        <td colspan="1"> 82.91</td>
        <td colspan="1"> 83.62 </td>
        <td colspan="1"> 83.75 </td>
        <td colspan="1"> 83.58 </td>
    </tr>
    </tr>
        <td colspan="1"> FedDC </td>
        <td colspan="1"> 83.15 </td>
        <td colspan="1"> 84.44 </td>
        <td colspan="1"> 84.51</td>
        <td colspan="1"> 84.63 </td>
        <td colspan="1"> 84.71 </td>
        <td colspan="1"> 84.66 </td>
    </tr>
    </tr>
        <td colspan="1"> FedDisco </td>
        <td colspan="1"> 81.25 </td>
        <td colspan="1"> 83.09 </td>
        <td colspan="1"> 83.11</td>
        <td colspan="1"> 83.75 </td>
        <td colspan="1"> 83.88 </td>
        <td colspan="1"> 83.73 </td>
    </tr>
    </tr>
        <td colspan="1"> FedVRA </td>
        <td colspan="1"> 81.33 </td>
        <td colspan="1"> 83.05 </td>
        <td colspan="1"> 83.25</td>
        <td colspan="1"> 83.98 </td>
        <td colspan="1"> 84.04 </td>
        <td colspan="1"> 84.01 </td>
    </tr>
    </tr>
        <td colspan="1"> FedGKD_PD </td>
        <td colspan="1"> 81.27 </td>
        <td colspan="1"> 82.98 </td>
        <td colspan="1"> 83.14</td>
        <td colspan="1"> 83.76 </td>
        <td colspan="1"> 83.92 </td>
        <td colspan="1"> 84.03 </td>
    </tr>
    </tr>
        <td colspan="1"> A_FedPD </td>
        <td colspan="1"> 83.25 </td>
        <td colspan="1"> 84.88 </td>
        <td colspan="1"> 84.95 </td>
        <td colspan="1"> 85.23 </td>
        <td colspan="1"> 85.20 </td>
        <td colspan="1"> 85.51 </td>
    </tr>
    </tr>
        <td colspan="1"> FedCADS </td>
        <td colspan="1"> 83.34 </td>
        <td colspan="1"> 85.05 </td>
        <td colspan="1"> 85.42 </td>
        <td colspan="1"> 85.58 </td>
        <td colspan="1"> 85.77 </td>
        <td colspan="1"> 85.80 </td>
    </tr>
        </tbody>
</table>
</p>
