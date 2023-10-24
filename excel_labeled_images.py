import pandas as pd


ref_assembly_csv = pd.read_csv('app/ref_assembly_with_variations_new.csv')
df = pd.read_csv('app/MT_merged_big_excel_set_with_variations.csv')
df_excel = ref_assembly_csv[ref_assembly_csv['Doc_ID'].isin(df['Doc_ID'])]
df_excel = df_excel.drop_duplicates('Doc_ID')
print()