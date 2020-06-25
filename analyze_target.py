import pandas as pd

# Sort the target data based on X and Y
target = pd.read_csv('train_target.csv')
target = target.sort_values(['X', 'Y', 'M', 'V'], ascending=[True, True, True, True])
print(target.shape)
print(target)


# Count the number of groups having same X, Y
groups = target.groupby(['X','Y'])
groups_count = len(groups)
print('Total groups number: {}'.format(groups_count))
for group_name, group in groups:
    print('{} group size: {}'.format(group_name, len(group)))
group_size = 35

new_target = target.copy()
new_target = new_target.drop(new_target.index[0:])

for set in range(group_size):
    for group_name, group in groups:
        new_target = new_target.append(group.apply(lambda x: x.iloc[set]), ignore_index=True)

print('\n', new_target.shape)
print(new_target)
new_target['id'] = new_target['id'].astype(int)
new_target.to_csv('new_train_target.csv', index=False)


# Transform the features data corresponding to the sequence of target data
features = pd.read_csv('train_features.csv', dtype=str)
features['id'] = features['id'].astype(int)

new_features = features.copy()
new_features = new_features.drop(new_features.index[0:])

features = features.groupby('id')
for i, row in new_target.iterrows():
    selected_group = features.get_group(row['id'])
    new_features = new_features.append(selected_group, sort=False)

print(new_features.shape)
print(new_features)
new_features.to_csv('new_train_features.csv', index=False)
