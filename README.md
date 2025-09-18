# FedCADS: Robust Federated Learning via Dual Distillation and Participation-Aware Optimization under Non-IID Data
Code for paper - **[FedCADS: Robust Federated Learning via Dual Distillation and Participation-Aware Optimization under Non-IID Data]**

We provide code to run FedCADS,
[A_FedPD](https://openreview.net/pdf?id=h1iMVi2iEM),
[FedGKD + PD](https://www.researchgate.net/profile/Dezhong-Yao/publication/373957735_FedGKD_Towards_Heterogeneous_Federated_Learning_via_Global_Knowledge_Distillation/links/65fae109a4857c7962653369/FedGKD-Toward-Heterogeneous-Federated-Learning-via-Global-Knowledge-Distillation.pdf),
[FedVRA](https://ojs.aaai.org/index.php/AAAI/article/view/26212),
[FedDC](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf), 
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf), 
[FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), 
[Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), and [FedFLD](https://ieeexplore.ieee.org/document/10887961) methods.


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
