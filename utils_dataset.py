import numpy as np
import torch.autograd


from utils_libs import *
from PIL import Image
from torch.utils.data import ConcatDataset



class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset = dataset
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" % (self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' % unbalanced_sgm if unbalanced_sgm != 0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' % (self.data_path, self.name)):
            # Get Raw data                

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                     std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                      train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                      train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;

            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                       train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                       train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;






            trn_itr = trn_load.__iter__();
            tst_itr = tst_load.__iter__()
            # labels are of shape (n_data,)
            trn_x, trn_y = trn_itr.__next__()
            tst_x, tst_y = tst_itr.__next__()

            trn_x = trn_x.numpy();
            trn_y = trn_y.numpy().reshape(-1, 1)
            tst_x = tst_x.numpy();
            tst_y = tst_y.numpy().reshape(-1, 1)

            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            # Draw from lognormal distribution
            clnt_data_list = (
                np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
            clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)

            # Add/Subtract the excess number starting from first client
            if diff != 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            ###     

            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_client)
                    # np.save("results/heterogeneity_distribution_{:s}.npy".format(self.dataset), cls_priors)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client, self.n_cls])
                    a[:, c::] = 0
                    [np.random.shuffle(i) for i in a]
                    # np.save("results/heterogeneity_distribution_{:s}_{:s}.npy".format(self.dataset, self.rule), a/c)
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i, j] != 0:
                                prior_cumsum[i, j] = a[i, 0:j + 1].sum() / c * 1.0

                idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                # print(cls_amount)
                clnt_x = [
                    np.zeros((clnt_data_list[client__], self.channels, self.height, self.width)).astype(np.float32)
                    for client__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[client__], 1)).astype(np.int64) for client__ in
                          range(self.n_client)]

                while (np.sum(clnt_data_list) != 0):
                    curr_client = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    # print('Remaining Data: %d' %np.sum(client_data_list))
                    if clnt_data_list[curr_client] <= 0:
                        continue
                    clnt_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1

                        clnt_x[curr_client][clnt_data_list[curr_client]] = trn_x[
                            idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_client][clnt_data_list[curr_client]] = trn_y[
                            idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                # clnt_x = np.asarray(clnt_x,dtype=object)
                # clnt_y = np.asarray(clnt_y,dtype=object)
                clnt_x = np.asarray(clnt_x, dtype='float32')
                clnt_y = np.asarray(clnt_y, dtype='float32')

                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt, cls] = np.mean(clnt_y[clnt] == cls)
                # prior_real_diff = np.abs(cls_means - cls_priors)
                # print('--- Max deviation from prior: %.4f' % np.max(prior_real_diff))
                # print('--- Min deviation from prior: %.4f' % np.min(prior_real_diff))

            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm == 0:
                assert len(trn_y) // 100 % self.n_client == 0

                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx]  # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client):  # devices
                    for j in range(100):  # class
                        clnt_x[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :, :, :] = trn_x[
                                                                                                            500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                        i + 1),
                                                                                                            :, :, :]
                        clnt_y[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :] = trn_y[
                                                                                                      500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                  i + 1),
                                                                                                      :]


            elif self.rule == 'iid':

                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]

                clnt_x = np.asarray(clnt_x,dtype='float32')
                clnt_y = np.asarray(clnt_y,dtype='float32')

            self.clnt_x = clnt_x
            self.clnt_y = clnt_y

            self.tst_x = tst_x
            self.tst_y = tst_y

            # Save data
            os.mkdir('%sData/%s' % (self.data_path, self.name))

            np.save('%sData/%s/clnt_x.npy' % (self.data_path, self.name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' % (self.data_path, self.name), clnt_y)

            np.save('%sData/%s/tst_x.npy' % (self.data_path, self.name), tst_x)
            np.save('%sData/%s/tst_y.npy' % (self.data_path, self.name), tst_y)

        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' % (self.data_path, self.name), allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x = np.load('%sData/%s/tst_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.tst_y = np.load('%sData/%s/tst_y.npy' % (self.data_path, self.name), allow_pickle=True)


            if self.dataset == 'CIFAR10':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;

        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % self.tst_y.shape[0])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name

        if self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if (np.random.rand() > .5):
                    # Random cropping
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y


class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)
