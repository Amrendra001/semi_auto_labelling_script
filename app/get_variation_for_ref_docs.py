import os
import boto3
import pandas as pd

from utils import get_file_name_from_path, download_vision_response



def apply_line_agg(x):
    return {
        'word_id': x['line_id'].iloc[0],
        'word': x.sort_values('minx')['word'].str.cat(sep=' '),
        'minx': x['minx'].min(),
        'maxx': x['maxx'].max(),
        'miny': x['miny'].min(),
        'maxy': x['maxy'].max(),
        'page': x['page'].iloc[0]
    }


def get_line_df_from_word_df(wdf):
    line_df = pd.DataFrame(wdf.groupby('line_id').apply(apply_line_agg).to_list())
    return line_df


def get_variation_for_doc(ocr_data, variation_csv, key_acc_id, doc_id):
    ocr_data  = get_line_df_from_word_df(ocr_data)
    lines = ocr_data['word'].astype(str).to_list()
    # print(lines)

    doc_ext = doc_id.split('.')[-1]
    
    # Filter by key account ID
    variation_csv_ka = variation_csv[variation_csv['Key_Account_ID']==key_acc_id].reset_index(drop=True)

    variation_csv_dt = pd.DataFrame(columns=variation_csv_ka.columns)
    # Filter by doc type (extension from doc id)
    for idx in range(len(variation_csv_ka)):
        row = variation_csv_ka.iloc[idx]
        if doc_ext.lower() in row['PO_Type'].lower().split(','):
            variation_csv_dt.loc[len(variation_csv_dt)] = row

    # variation_csv_dt = variation_csv_ka[doc_ext.lower() in variation_csv_ka['PO_Type'].lower()]
    possible_variation_by_doc_type_df = variation_csv_dt # filter df by "document_type in config['poType'].lower().split(',')"

    # Filter by variation string search    
    if len(possible_variation_by_doc_type_df)==1:
        final_variation = possible_variation_by_doc_type_df['Variation'][0]
    elif len(possible_variation_by_doc_type_df)>1:
        final_variation = 1
        for var_idx in range(len(possible_variation_by_doc_type_df)): 
            # string=row['Variation_String']
            row = possible_variation_by_doc_type_df.iloc[var_idx]
            string=row['Variation_String']
            if string!=None:
                res = [i for i in lines if row['Variation_String'] in i]
                if len(res)>0:
                    final_variation=row['Variation']
                    break
    else:
        final_variation = -1

    return final_variation


# def get_variation_for_input_df(input_csv_name):
#     os.environ['DATA_BUCKET'] = 'javis-ai-parser-dev'

#     input_csv = pd.read_csv(input_csv_name)

#     variation_csv = pd.read_csv('key_acc_variations.csv')

#     s3_client = boto3.client('s3')

#     input_csv['Variation'] = ''
#     input_list = []
#     input_idxs = []

#     for idx in tqdm(range(len(input_csv))):
#         row = input_csv.iloc[idx]
#         doc_id = row['Doc_ID']
#         key_acc_id = row['Javis_DB_ID']

#         doc_id_without_ext = get_file_name_from_path(doc_id)

#         if os.path.isfile(f'/tmp/{doc_id_without_ext}.parquet'):
#             wdf = pd.read_parquet(f'/tmp/{doc_id_without_ext}.parquet')
#         else:
#             try:
#                 wdf= download_vision_response(doc_id_without_ext,s3_client)
#             except:
#                 print(f'OCR data not downloaded for doc: {doc_id}')
#                 continue

#         ldf = get_line_df_from_word_df(wdf)

#         input_tuple = (ldf, variation_csv, key_acc_id, doc_id)
#         input_list.append(input_tuple)
#         input_idxs.append(idx)

#         # variation = get_variation_for_doc(ldf, variation_csv, key_acc_id, doc_id)
    
#     output_list = multiprocessing_handler(get_variation_for_doc,input_list)

#     for i, (idx,variation) in enumerate(zip(input_idxs,output_list)):
#         key_acc_id = input_list[i][2]
#         input_csv.at[idx, 'Variation'] = str(key_acc_id) + '_' + str(variation)

#     input_csv.to_csv('.'.join(input_csv_name.split('.')[:-1])+'_with_variations.csv')



# if __name__=='__main__':
#     input_csv_name = 'MT_7k_set_assembly.csv'
#     get_variation_for_input_df(input_csv_name)
