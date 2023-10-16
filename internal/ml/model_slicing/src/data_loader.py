import glob
from tqdm import tqdm
import torch
import random
from torch.utils.data import Dataset, DataLoader,Subset
import os


def decode_libsvm(line):
    columns = line.split(' ')
    def map_func(pair): return (int(pair[0]), float(pair[1]))
    id, value = zip(
        *map(lambda col: map_func(col.split(':')), columns[1:]))
    sample = {'id': torch.LongTensor(id),
                'value': torch.FloatTensor(value),
                'y': float(columns[0])}
    return sample

class SQLAttacedLibsvmDataset(Dataset):
    """ Dataset loader for Libsvm data format """

    def __init__(self, fname, nfields, max_filter_col):

        with open(fname) as f:
            sample_lines = sum(1 for line in f)

        self.generate_sql = True if max_filter_col != 0 else False

        self.feat_id = torch.LongTensor(sample_lines, nfields)
        self.feat_value = torch.FloatTensor(sample_lines, nfields)
        self.y = torch.FloatTensor(sample_lines)

        self.nsamples = 0
        with tqdm(total=sample_lines) as pbar:
            with open(fname) as fp:
                line = fp.readline()
                while line:
                    try:
                        sample = decode_libsvm(line)
                        self.feat_id[self.nsamples] = sample['id']
                        self.feat_value[self.nsamples] = sample['value']
                        self.y[self.nsamples] = sample['y']
                        self.nsamples += 1
                    except Exception:
                        print(f'incorrect data format line "{line}" !')
                    line = fp.readline()
                    pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')

        # generate the columsn statics
        self.max_columns = min(self.feat_id.shape[1], max_filter_col)

        # number of columns
        self.ncols = self.feat_id.shape[1]

        # Convert the tensor to a list of lists, where each inner list contains the unique values from each column
        self.col_cardinalities = [
            self.feat_id[:, i].unique().tolist() for i in range(self.ncols)]

        # add pedding feature_id to each of the columns unique value.
        self.padding_feature_id = []
        max_value = max(
            value for sublist in self.col_cardinalities for value in sublist)
        for i, sublist in enumerate(self.col_cardinalities):
            # in place append
            sublist.append(max_value + 1 + i)
            self.padding_feature_id.append(max_value + 1 + i)

        print("self.sql_history is initialized !")
        # this is used in infernece, as infernece must testd on the trained sql.
        self.sql_history = set()

    def generate_sql_by_row(self, row: torch.Tensor):
        # 1. firstly randomly pick number of the columns
        ncol = random.randint(0, self.max_columns)

        # default to not select any value, init all columns feature id to last of each list
        random_sql = [col[-1] for col in self.col_cardinalities]

        # 2. second, randomly pick two cols,
        selected_cols = random.sample(range(len(random_sql)), ncol)

        # 3. assign value from row to selected columns in random_sql
        for col in selected_cols:
            random_sql[col] = row[col].item()   # this is feature id

        # record history
        self.sql_history.add(tuple(random_sql))
        return torch.tensor(random_sql)
    
    def generate_default_sql(self):
        # default to not select any value, init all columns feature id to last of each list
        random_sql = [col[-1] for col in self.col_cardinalities]
        return torch.tensor(random_sql)
    
    
    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'sql': self.generate_sql_by_row(self.feat_id[idx]),
                'default_sql':self.generate_default_sql(),
                'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}


    def reset_sql_his(self):
        """
        Called after each epoch. such that infernece can use the same sql set.
        :return:
        """
        self.sql_history.clear()


class SQLAwareDataset(Dataset):
    
    def __init__(self, fname, nfields):
        with open(fname) as f:
            sample_lines = sum(1 for line in f)
            
        self.feat_id = torch.LongTensor(sample_lines, nfields)
        self.feat_value = torch.FloatTensor(sample_lines, nfields)
        self.y = torch.FloatTensor(sample_lines)
        
        self.nsamples = 0
        
        with tqdm(total=sample_lines) as pbar:
            with open(fname) as fp:
                line = fp.readline()
                while line:
                    try:
                        sample = decode_libsvm(line)
                        self.feat_id[self.nsamples] = sample['id']
                        self.feat_value[self.nsamples] = sample['value']
                        self.y[self.nsamples] = sample['y']
                        self.nsamples += 1
                    except Exception:
                        print(f'incorrect data format line "{line}" !')
                    line = fp.readline()
                    pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')
        
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}
        
class Workload(object):
    
    def __init__(self, dataset:Dataset, dir_path:str):
        print(dir_path)
        assert os.path.exists(dir_path)
        
        sql_file = os.path.join(dir_path, "sql.txt")
        idx_file_dir = os.path.join(dir_path, "data_idx")
        
        assert os.path.exists(sql_file)
        assert os.path.exists(idx_file_dir)
        
        self.dataset = dataset
        self.init_sql(file_path=sql_file)
        
        self.sub_idx_path_list = [os.path.join(idx_file_dir, i) for i in os.listdir(idx_file_dir)]

        self.length = len(self.sql_list)
        assert self.length == len(self.sub_idx_path_list)
        
        
    def init_sql(self, file_path:str):
        self.sql_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                self.sql_list.append(eval(line.split(":")[-1].strip()))

    def __len__(self):
        return self.length
    
    
    def __getitem__(self, index):
        with open(self.sub_idx_path_list[index], 'r') as f:
            idxs = eval(f.read().strip())
            
        sql = torch.LongTensor(self.sql_list[index])
        return Subset(self.dataset, idxs), sql
    
    


def sql_attached_dataloader(args):
    
    data_dir = os.path.join(args.data_dir, args.dataset)
    train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]
    val_file = glob.glob("%s/va*libsvm" % data_dir)[0]
    test_file = glob.glob("%s/te*libsvm" % data_dir)[0]

    train_loader = DataLoader(SQLAttacedLibsvmDataset(train_file, args.nfield, args.max_filter_col),
                              batch_size=args.batch_size, shuffle=True,
                              pin_memory=True)
    
    val_loader = DataLoader(SQLAttacedLibsvmDataset(val_file, args.nfield, args.max_filter_col),
                              batch_size=args.batch_size, shuffle=False,
                              pin_memory=True)
    
    # val_dataset = SQLAwareDataset(val_file, args.nfield)
    # val_workload = Workload(val_dataset, os.path.join(data_dir, args.workload))
    
    test_dataset = SQLAwareDataset(test_file, args.nfield)
    # print(data_dir)
    # print(args.workload)
    workload_dir = os.path.join(data_dir, "workload", args.workload)

    test_workload = Workload(test_dataset, workload_dir)
    
    return train_loader, val_loader, test_workload


''' ---------------------  Big Dataset Loading ---------------------'''

def load_data(data_dir, namespace):
    print(f'# loading data from '
          f'{data_dir}/{namespace}_feat_id.pt, '
          f'{data_dir}/{namespace}_feat_value.pt'
          f'{data_dir}/{namespace}_y.pt ......')

    feat_id = torch.load(f'{data_dir}/{namespace}_feat_id.pt')
    feat_value = torch.load(f'{data_dir}/{namespace}_feat_value.pt')
    y = torch.load(f'{data_dir}/{namespace}_y.pt')

    print(f'# {int(y.shape[0])} data samples loaded...')

    return feat_id, feat_value, y, int(y.shape[0])


class LibsvmDatasetReadOnce(Dataset):
    """ Dataset loader for Libsvm data format """

    def __init__(self, fname):
        parent_directory = os.path.dirname(fname)
        if "train" in fname:
            namespace = "decoded_train"
        elif "valid" in fname:
            namespace = "decoded_valid"
        else:
            raise
        self.feat_id, self.feat_value, self.y, self.nsamples = load_data(
            parent_directory, namespace)

        print(f'# {self.nsamples} data samples loaded...')

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}


def devoded_libsvm_dataloader(args, data_dir, batch_size):
    print("Loading data from ", data_dir)
    workers = args.workers
    train_file_name = f"{data_dir}/train.libsvm"
    valid_file_name = f"{data_dir}/valid.libsvm"
    test_file_name = f"{data_dir}/test.libsvm"
    print(f"using train={train_file_name}, valid={valid_file_name}")
    # read the converted file
    if args.device == "cpu":
        train_loader = DataLoader(LibsvmDatasetReadOnce(train_file_name),
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(LibsvmDatasetReadOnce(valid_file_name),
                                batch_size=batch_size * 8,
                                shuffle=False)

    else:
        train_loader = DataLoader(LibsvmDatasetReadOnce(train_file_name),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  pin_memory=True)

        val_loader = DataLoader(LibsvmDatasetReadOnce(valid_file_name),
                                batch_size=batch_size * 8,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True)

    return train_loader, val_loader
