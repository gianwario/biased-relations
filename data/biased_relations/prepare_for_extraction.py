import pandas as pd

# Load the CSV file
df = pd.read_csv('crowspairs/crows_pairs_anonymized.csv')

# For each unique bias_type, save a separate CSV
for bias in df['bias_type'].unique():
    df_subset = df[df['bias_type'] == bias]
    
    #save filtered
    filename = f'{bias}_bias.csv'
    df_subset.to_csv(filename, index=False)

    # Filter rows where stereo_antistereo == "stereo"
    df_subset = df_subset[df_subset['stereo_antistereo'] == 'stereo']
    # keep only sentence
    df_subset = df_subset.drop(columns=['Unnamed: 0', 'bias_type', 'sent_less', 'stereo_antistereo', 'annotations', 'anon_writer', 'anon_annotators'])

    #save only more stereotyped sentences
    filename = f'stereo_sentences_{bias}_bias.csv'
    df_subset.to_csv(filename, index=False)

   

print("CSV files created for each bias_type with stereo examples only.")