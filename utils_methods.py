import copy

import numpy as np
import torch

from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
from tensorboardX import SummaryWriter
from collections import Counter

### Methods
def train_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch, 
                                     com_amount, print_per, weight_decay, 
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedAvg_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)
   
    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
        
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedAvg/%s/%s' %(data_path, data_obj.name, suffix[:26]))
    
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i
                
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))

    
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):         
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
            
            
                clnt_models[clnt] = model_func().to(device)
                
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model_avg(clnt_models[clnt], trn_x, trn_y,
                                                tst_x, tst_y, 
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay, 
                                                data_obj.dataset, sch_step, sch_gamma)
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))

            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )
                        
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                
                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                
    return fed_mdls_sel, tst_perf_sel

def train_FedProx(data_obj, act_prob ,learning_rate, batch_size, epoch, 
                                     com_amount, print_per, weight_decay,
                                     model_func, init_model, alpha_coef, sch_step, sch_gamma,
                                     save_period, weight_uniform = False, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedProx_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay,alpha_coef)

    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed
    suffix += '_WU%s' %(weight_uniform)

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

        
    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
        
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedProx/%s/%s' %(data_path, data_obj.name, suffix[:26]))
    
    
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model
                


               
                
                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))

    
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):         
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
            cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
        
          
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness

            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
                # Add regulariser during training
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #local train
                clnt_models[clnt] = train_model_prox(clnt_models[clnt],cld_mdl_param_tensor, alpha_coef, trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i),
                                                batch_size, epoch, print_per,
                                                weight_decay, 
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            cld_mdl_param= np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )            

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                
                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                
    return fed_mdls_sel, tst_perf_sel


def train_SCAFFOLD(data_obj, act_prob ,learning_rate, batch_size, n_minibatch, 
                                     com_amount, print_per, weight_decay, 
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1, global_learning_rate=1):
    suffix = 'Scaffold_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)
   
    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it
        
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_Scaffold/%s/%s' %(data_path, data_obj.name, suffix[:26]))
    
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                
                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))                    # Get state_params_diffs
                    state_params_diffs = np.load('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):         
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
          
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            del clnt_models
            
            clnt_models = list(range(n_clnt))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

            
            
                clnt_models[clnt] = model_func().to(device)
                
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                    
                # Scale down c
                state_params_diff_curr = torch.tensor(-state_params_diffs[clnt] + state_params_diffs[-1]/weight_list[clnt], dtype=torch.float32, device=device)
                
                clnt_models[clnt] = train_scaffold_mdl(clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, 
                    learning_rate * (lr_decay_per_round ** i), batch_size, n_minibatch, print_per,
                    weight_decay, data_obj.dataset, sch_step, sch_gamma)
                
                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_params_diffs[clnt] - state_params_diffs[-1] + 1/n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_params_diffs[clnt])*weight_list[clnt]
                state_params_diffs[clnt] = new_c
                
                clnt_params_list[clnt] = curr_model_param
                
            
            avg_model_params = global_learning_rate*np.mean(clnt_params_list[selected_clnts], axis = 0) + (1-global_learning_rate)*prev_params
           
            avg_model = set_client_from_params(model_func().to(device), avg_model_params) 
            
            state_params_diffs[-1] += 1 / n_clnt * delta_c_sum

            
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))

            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )
            
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), state_params_diffs)
                
                
                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1-save_period)) 
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                
    return fed_mdls_sel, tst_perf_sel


def train_FedDyn(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, print_per, 
                  weight_decay,  model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, save_period,
                  suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix  = 'FedDyn_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' %rand_seed
    suffix += '_lrdecay%f' %lr_decay_per_round
    
    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances)) # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))
    
    n_par = len(get_mdl_params([model_func()])[0])
    
    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1
            
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDyn/%s/%s' %(data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                               %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' %(data_path, data_obj.name, suffix, i+1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr//save_period] = fed_cld 
                
                if os.path.exists('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1))):

                    tst_cur_cld_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))
                    
                                   
                      
    if (trial) or (not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' %(data_path, data_obj.name, suffix, com_amount))):
        
        
        if saved_itr == -1:
            
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        
        else:            
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        
    
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist    
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            
            del clnt_models
            clnt_models = list(range(n_clnt))
            
            for clnt in selected_clnts:
                # Train locally 
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                
                clnt_models[clnt] = model_func().to(device)
                
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedDyn(model, model_func, alpha_coef_adpt,
                                                     cld_mdl_param_tensor, hist_params_diffs_curr, 
                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), 
                                                     batch_size, epoch, print_per, weight_decay, 
                                                     data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par
                        
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis = 0)
            
            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)
            
            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 

            
            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]
            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Current cloud':tst_cur_cld_perf[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Current cloud':tst_cur_cld_perf[i][1]
                   }, i
                  )     
            
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt' 
                           %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_cur_cld_perf[:i+1])
                    
                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                     

                if (i+1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                    
                    
            if ((i+1) % save_period == 0):
                avg_cld_mdls[i//save_period] = cur_cld_model
                    
    return avg_cld_mdls, tst_cur_cld_perf



def train_FedDC(data_obj, act_prob, n_minibatch,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, init_model, alpha_coef,
                sch_step, sch_gamma, save_period,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedDC_' + str(act_prob) + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)

    avg_cld_mdls = list(range(n_save_instances))

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    ###
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32')  # including cloud state

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDC/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i


                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            delta_g_sum = np.zeros(n_par)

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                local_update_last = state_gadient_diffs[clnt]  # delta theta_i
                global_update_last = state_gadient_diffs[-1] / weight_list[clnt]  # delta theta
                alpha = alpha_coef / weight_list[clnt]
                hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                clnt_models[clnt] = train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last,
                                                      global_mdl, hist_i,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, print_per, weight_decay, data_obj.dataset,
                                                      sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += delta_param_curr
                beta = 1 / n_minibatch / learning_rate

                state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
                delta_g_cur = (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
                delta_g_sum += delta_g_cur
                state_gadient_diffs[clnt] = state_g
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            delta_g_cur = 1 / n_clnt * delta_g_sum
            state_gadient_diffs[-1] += delta_g_cur

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####


            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved array
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf



def train_FedDisco(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedDisco' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDisco/%s/%s' % (data_path, data_obj.name, suffix[:28]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedDyn(model, model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift
                clnt_params_list[clnt] = curr_model_par

            total_data_points = 0
            for clnt in selected_clnts:
                total_data_points += data_obj.clnt_y[clnt].shape[0]

            fed_avg_freqs = [data_obj.clnt_y[clnt].shape[0] / total_data_points for clnt in selected_clnts]

            traindata_cls_counts = np.asarray(
                [np.mean(data_obj.clnt_y[clnt] == cls) for clnt in range(data_obj.n_client) for cls in
                 range(data_obj.n_cls)]).reshape(-1, 10)
            for clnt in range(data_obj.n_client):
                traindata_cls_counts[clnt] = np.multiply(traindata_cls_counts[clnt], data_obj.clnt_y[clnt].shape[0])

            global_dist = np.ones(traindata_cls_counts.shape[1]) / traindata_cls_counts.shape[1]

            # avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            distribution_difference = get_distribution_difference(traindata_cls_counts, selected_clnts, 'kl', global_dist)
            fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs,distribution_difference,0.5,0.1)
            avg_mdl_param_sel = np.dot(np.array(fed_avg_freqs).reshape(1, -1), clnt_params_list[selected_clnts])
            avg_mdl_param_sel = np.squeeze(avg_mdl_param_sel)
            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0) #1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



            #####



            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf




def train_Fedvra(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedVRA_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y


    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)

    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    lamda = torch.from_numpy(np.zeros((n_clnt, n_par)).astype('float32')).to(device)
    lamda_num = torch.from_numpy(np.zeros(n_clnt).astype('float32')).to(device)

    corr_palty = 0.1
    a_step_size = 7

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedVRA/%s/%s' % (data_path, data_obj.name, suffix[:24]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                # update lamda
                lamda[clnt] += corr_palty * a_step_size * torch.from_numpy(hist_params_diffs[clnt]).to(device)
                lamda_num[clnt]+=1
                lamda[clnt]=lamda[clnt]/lamda_num[clnt]


                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_Fedvra(model, model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma,lamda[clnt])
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabmla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par


            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)




            #####


            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf





def train_FedVARP(data_obj, act_prob, learning_rate, batch_size, n_minibatch,
                  com_amount, print_per, weight_decay,
                  model_func, init_model, sch_step, sch_gamma,
                  save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                  global_learning_rate=1):
    suffix = 'FedVARP_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y




    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt  # normalize it

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params = np.zeros((n_clnt + 1, n_par)).astype('float32')
    delta_params = np.zeros((n_clnt,n_par)).astype('float32')
    # state_params_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32')  # including cloud state
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedVARP/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):


                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1))  # Get state_params_diffs
                    state_params = np.load(
                        '%sModel/%s/%s/%d_state_params.npy' % (data_path, data_obj.name, suffix, i + 1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models

            clnt_models = list(range(n_clnt))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0] # global params
            global_mdl = torch.tensor(prev_params, dtype=torch.float32, device=device)  # Theta

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True

                clnt_models[clnt] = train_fedvarp_mdl(clnt_models[clnt], model_func, trn_x,
                                                       trn_y,
                                                       learning_rate * (lr_decay_per_round ** i), batch_size,
                                                       n_minibatch, print_per,
                                                       weight_decay, data_obj.dataset, sch_step, sch_gamma)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]  #w_i^(t,tau)
                delta_params[clnt] = (prev_params - curr_model_param) / n_minibatch / learning_rate
                clnt_params_list[clnt] = curr_model_param
            for clnt in unselected_clnts:
                delta_params[clnt] = state_params[clnt]
            delta_global_params = state_params[-1] + np.mean((delta_params[selected_clnts] - state_params[selected_clnts]), axis=0)
            avg_model_params = prev_params - learning_rate * n_minibatch * delta_global_params
            avg_model = set_client_from_params(model_func().to(device), avg_model_params)

            state_params[-1] += np.mean((delta_params - state_params[:-1]),axis=0)
            state_params[:-1] = delta_params[:]


            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        state_params)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%d_state_params.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_A_FedPD(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,rho,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'A_FedPD_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_A_FedPD/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_A_FedPD(model, model_func, alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
               
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)   # \overline{\theta^{t+1}} = \frac{1}{P}\sum_{i \in P_t} \theta_i^{t+1}
            for clnt in unselected_clnts:
                hist_params_diffs[clnt] += avg_mdl_param_sel - cld_mdl_param
            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf





def train_FedGKD(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedGKD_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedGKD/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedGkd(model, model_func, cur_cld_model,alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf









def train_FedCADS(data_obj, rho, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedCADS_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDyn/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedCADS(i, com_amount, model, cur_cld_model, model_func, alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += rho * (curr_model_par - cld_mdl_param)
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf


