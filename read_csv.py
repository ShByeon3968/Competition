import pandas as pd

csv_file = pd.read_csv('./data/sample_submission.csv')

print(csv_file.keys())