import torch.utils.data as data
import torch
dataset = torch.zeros((10,10))
dataset[:, 0] = torch.arange(10)
print(dataset)


# Sebset Sampler
print('\n--------Subset Sampler----------')
data_loader = data.DataLoader(dataset,
                              batch_size=4,
                              sampler = data.SubsetRandomSampler(torch.arange(8)))

data_iter = iter(data_loader)
for i, batch in enumerate(data_loader, 1):
    print(f'Batch Index: {i}')
    print(f'Batch Data: {batch}')


# Replacement = True for 20 samplers
print('\n--------Replacement = True for 20 samplers----------')
data_loader = data.DataLoader(dataset,
                              batch_size=5,
                              sampler = data.RandomSampler(torch.arange(10), replacement=True, num_samples=20))

data_iter = iter(data_loader)
for i, batch in enumerate(data_loader, 1):
    print(f'Batch Index: {i}')
    print(f'Batch Data: {batch}')

# Prob sampler for unbalanced data
# [0, 1, 2, 3, 4, 5, 6, 6] for cat image, [7, 8, 9] for dog image
# 3/10, 3/10, 3/10, 3/10, 3/10, 3/10, 3/10, 7/10, 7/10, 7/10
probs = torch.cat([torch.ones(7,) * 3/10, torch.ones(3,) * 7/10])
print('\n--------Weighted Sampler----------')
data_loader = data.DataLoader(dataset,
                              batch_size=5,
                              sampler = data.WeightedRandomSampler(probs, replacement=True, num_samples=20))
data_iter = iter(data_loader)
for i, batch in enumerate(data_loader, 1):
    print(f'Batch Index: {i}')
    print(f'Batch Data: {batch}')