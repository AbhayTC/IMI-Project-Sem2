import pandas as pd
import pickle
import os

os.makedirs('outputs', exist_ok=True)

print('Attempting to recover master_dataset.pkl to outputs/master_dataset.csv...')
if os.path.exists('master_dataset.pkl'):
    with open('master_dataset.pkl', 'rb') as f:
        df = pickle.load(f)
    print(f'Successfully loaded PKL with {df.shape[1]} columns. Exporting...')
    df.to_csv('outputs/master_dataset.csv', index=False)
    print('Recovered outputs/master_dataset.csv!')
else:
    print('master_dataset.pkl not found in the root directory!')

print('\nAttempting to recover morgan_fingerprints.pkl to outputs/morgan_fingerprints.csv...')
if os.path.exists('morgan_fingerprints.pkl'):
    with open('morgan_fingerprints.pkl', 'rb') as f:
        df2 = pickle.load(f)
    print(f'Successfully loaded PKL with {df2.shape[1]} columns. Exporting...')
    df2.to_csv('outputs/morgan_fingerprints.csv', index=False)
    print('Recovered outputs/morgan_fingerprints.csv!')
else:
    print('morgan_fingerprints.pkl not found in the root directory!')
