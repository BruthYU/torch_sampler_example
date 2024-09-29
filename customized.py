import torch.utils.data as data
import torch
import numpy as np
# dataset
dataset = torch.zeros((10,10))
dataset[:, 0] = torch.arange(10)
batch_size = 2
print(dataset)
print(f'\n--------Batch Size: {batch_size}----------')


class MySampler(data.Sampler):
    def __init__(self, dataset, num_samples):
        self.num_samples = num_samples
        self.dataset = dataset
        self.batch_size = batch_size
        self._dataset_indices = list(range(len(dataset)))

    def __iter__(self):
        sample_indices = np.random.choice(self._dataset_indices, size=self.num_samples, replace=False).tolist()
        sorted_sample_indices = sorted(sample_indices)
        return iter(sorted_sample_indices)
        # yield from iter(sorted_sample_indices)

    def __len__(self):
        return self.num_samples


my_sampler = MySampler(dataset, num_samples=8)

data_loader = data.DataLoader(dataset,
                              batch_size=2,
                              sampler=my_sampler)


print('\n--------Customized Sampler----------')
data_iter = iter(data_loader)
for i, batch in enumerate(data_loader, 1):
    print(f'Batch Index: {i}')
    print(f'Batch Data: {batch}')







