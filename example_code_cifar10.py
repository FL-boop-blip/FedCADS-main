import torch

from utils_general import *
from utils_methods import *
from utils_methods_FedCADS import *
from test_gpu import check_available_gpus, set_device, check_cudnn

# Dataset initialization
data_path = 'Folder/'  # The folder to save Data & Model

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
#      - In non-IID use Dirichlet distribution with rule_arg=0.6 or 0.3, Pathological with rule_arg= 3 or 6 for CIFAR10 and rule_arg=10 or 20 for CIFAR100.
#########

if __name__ == '__main__':

    check_available_gpus()
    n_client = 100
    # Generate IID or Dirichlet distribution
    # IID
    data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, rule='Dirichlet',rule_arg=0.6,
                             data_path=data_path)

    # Dirichlet (0.6)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
    # Dirichlet (0.3)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)

    model_name = 'cifar10_LeNet_fusion' #model type for FedCADS

    # model_name = 'cifar10_LeNet' #model type for other methods

    ###
    # Common hyperparameters

    com_amount = 1600
    save_period = 1600
    weight_decay = 1e-3
    batch_size = 50
    act_prob = 0.02
    suffix = model_name
    lr_decay_per_round = 0.998

    # Model function
    model_func = lambda: client_model(model_name)
    init_model = model_func()
    

    # Initalise the model for all methods with a random seed or load it from a saved initial model
    torch.manual_seed(37)
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' % (data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' % (data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))




    print('FedCADS')

    epoch = 5
    alpha_coef = 0.1
    learning_rate = 0.1
    print_per = epoch // 2

    [avg_cld_mdls, tst_cur_cld_perf] = train_FedCADS(data_obj=data_obj, act_prob=act_prob,
                                                    learning_rate=learning_rate,
                                                    batch_size=batch_size, epoch=epoch,
                                                    com_amount=com_amount, print_per=print_per,
                                                    weight_decay=weight_decay,
                                                    model_func=model_func, init_model=init_model,
                                                    alpha_coef=alpha_coef,
                                                    sch_step=1, sch_gamma=1, save_period=save_period,
                                                    suffix=suffix, trial=False,
                                                    data_path=data_path,
                                                    lr_decay_per_round=lr_decay_per_round)

    

    # print('FedGKD')
    #
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedGKD(data_obj=data_obj, act_prob=act_prob,
    #                                                                   learning_rate=learning_rate,
    #                                                                   batch_size=batch_size, epoch=epoch,
    #                                                                   com_amount=com_amount, print_per=print_per,
    #                                                                   weight_decay=weight_decay,
    #                                                                   model_func=model_func, init_model=init_model,
    #                                                                   alpha_coef=alpha_coef,
    #                                                                   sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                                   suffix=suffix, trial=False,
    #                                                                   data_path=data_path,
    #                                                                   lr_decay_per_round=lr_decay_per_round)

   
    #
    # # print('FedVRA')
    # #
    # # epoch = 5
    # # alpha_coef = 0.1
    # # learning_rate = 0.1
    # # print_per = epoch // 2
    # #
    # # [avg_cld_mdls,  tst_cur_cld_perf] = train_FedAvr(data_obj=data_obj, act_prob=act_prob,
    # #                                                                   learning_rate=learning_rate,
    # #                                                                   batch_size=batch_size, epoch=epoch,
    # #                                                                   com_amount=com_amount, print_per=print_per,
    # #                                                                   weight_decay=weight_decay,
    # #                                                                   model_func=model_func, init_model=init_model,
    # #                                                                   alpha_coef=alpha_coef,
    # #                                                                   sch_step=1, sch_gamma=1, save_period=save_period,
    # #                                                                   suffix=suffix, trial=False,
    # #                                                                   data_path=data_path,
    # #                                                                   lr_decay_per_round=lr_decay_per_round)
    #
    
    #
    # print('FedDisco')
    #
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedDisco(data_obj=data_obj, act_prob=act_prob,
    #                                                   learning_rate=learning_rate,
    #                                                   batch_size=batch_size, epoch=epoch,
    #                                                   com_amount=com_amount, print_per=print_per,
    #                                                   weight_decay=weight_decay,
    #                                                   model_func=model_func, init_model=init_model,
    #                                                   alpha_coef=alpha_coef,
    #                                                   sch_step=1, sch_gamma=1,
    #                                                   save_period=save_period,
    #                                                   suffix=suffix, trial=False,
    #                                                   data_path=data_path,
    #                                                   lr_decay_per_round=lr_decay_per_round)
    #
    # print('FedDC')
    #
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    #
    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob,
    #                                                n_minibatch=n_minibatch,
    #                                                learning_rate=learning_rate, batch_size=batch_size,
    #                                                epoch=epoch,
    #                                                com_amount=com_amount, print_per=print_per,
    #                                                weight_decay=weight_decay,
    #                                                model_func=model_func, init_model=init_model,
    #                                                alpha_coef=alpha_coef,
    #                                                sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                suffix=suffix, trial=False,
    #                                                data_path=data_path,
    #                                                lr_decay_per_round=lr_decay_per_round)
    #
    # print('FedDyn')
    #
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
    #                                                 learning_rate=learning_rate,
    #                                                 batch_size=batch_size, epoch=epoch,
    #                                                 com_amount=com_amount, print_per=print_per,
    #                                                 weight_decay=weight_decay,
    #                                                 model_func=model_func, init_model=init_model,
    #                                                 alpha_coef=alpha_coef,
    #                                                 sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                 suffix=suffix, trial=False,
    #                                                 data_path=data_path,
    #                                                 lr_decay_per_round=lr_decay_per_round)

    
    # # # ###
    # print('SCAFFOLD')
    #
    # epoch = 5
    #
    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
    #
    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    # learning_rate = 0.1
    # print_per = 5
    #
    # [fed_mdls_sel, tst_perf_sel] = train_SCAFFOLD(
    #     data_obj=data_obj, act_prob=act_prob,
    #     learning_rate=learning_rate, batch_size=batch_size, n_minibatch=n_minibatch,
    #     com_amount=com_amount, print_per=n_minibatch // 2, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model,
    #     sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
    #     trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)
    #
    # ####
    # print('FedAvg')
    #
    # epoch = 5
    # learning_rate = 0.1
    # print_per = 5
    #
    # [fed_mdls_sel, tst_perf_sel] = train_FedAvg(
    #     data_obj=data_obj, act_prob=act_prob,
    #     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
    #     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model,
    #     sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
    #     trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)
    #
    # # ####
    # print('FedProx')
    #
    # epoch = 5
    # learning_rate = 0.1
    # print_per = 5
    # alpha_coef = 0.1
    #
    # [fed_mdls_sel, tst_perf_sel] = train_FedProx(
    #     data_obj=data_obj, act_prob=act_prob,
    #     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
    #     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model, sch_step=1, sch_gamma=1,
    #     save_period=save_period, alpha_coef=alpha_coef, suffix=suffix, trial=False,
    #     data_path=data_path, lr_decay_per_round=lr_decay_per_round)

    
    # print('A_FedPD')
    #
    # epoch = 5
    # alpha_coef = 0.1
    # rho = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_A_FedPD(data_obj=data_obj, act_prob=act_prob,
    #                                                  learning_rate=learning_rate,
    #                                                  batch_size=batch_size, epoch=epoch,
    #                                                  com_amount=com_amount, print_per=print_per,
    #                                                  weight_decay=weight_decay,
    #                                                  model_func=model_func, init_model=init_model,
    #                                                  alpha_coef=alpha_coef, rho=rho,
    #                                                  sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                  suffix=suffix, trial=False,
    #                                                  data_path=data_path,
    #                                                  lr_decay_per_round=lr_decay_per_round)

    