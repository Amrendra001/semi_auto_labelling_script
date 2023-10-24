import os
import json
import boto3
import pandas as pd
from label_from_ref import get_page_labels
from utils import get_file_name_from_path, sync_label_s3_folder, download_vision_response, filter_special_char_words
from get_variation_for_ref_docs import get_variation_for_doc


LABELLED_FILES_S3_PATH = os.environ['LABELLED_FILES_S3_PATH']
LOCAL_FINAL_LABELS= '/tmp/ie_semi_auto_labelling/final'
FINAL_LABELS_TARGET_S3_PATH = os.environ['FINAL_LABELS_TARGET_S3_PATH']
REF_FOLDER = '/tmp/ie_semi_auto_labelling/ref'


def lambda_handler(event, context):
    if "Records" in event:
        event = json.loads(event['Records'][0]['body'])

    pred_doc_id = event['doc_id']
    pred_key_acc_id = int(event['key_acc_id'])
    try:
        pred_template_id = int(event['template_id'])
    except:
        pred_template_id = -1
    # pred_doc_id_without_ext = get_file_name_from_path(pred_doc_id)
    pred_doc_id_without_ext = '.'.join(pred_doc_id.split('.')[:-1])

    sync_label_s3_folder(LABELLED_FILES_S3_PATH,REF_FOLDER)

    ref_assembly_csv = pd.read_csv('app/ref_assembly_with_variations_new.csv')
    variation_csv = pd.read_csv('app/key_acc_variations.csv')

    s3_client = boto3.client('s3')
    pred_ocr = download_vision_response(pred_doc_id_without_ext, s3_client)
    pred_ocr = filter_special_char_words(pred_ocr)

    pred_variation = get_variation_for_doc(pred_ocr, variation_csv, pred_key_acc_id, pred_doc_id)

    ref_doc_list = []

    pred_variation = str(pred_key_acc_id) + '_' + str(pred_variation)

    #Search By Template ID First:
    ref_assembly_filt = ref_assembly_csv[ref_assembly_csv['Template_ID']==str(pred_template_id)]
    
    # 2nd Fallback
    if len(ref_assembly_filt)==0:
        ref_assembly_filt = ref_assembly_csv[ref_assembly_csv['Variation']==pred_variation]

    # 3rd Fallback Hardcode:
    # if len(ref_assembly_filt)==0 and pred_key_acc_id>1200:
    #     pred_template_id = 6629
    #     ref_assembly_filt = ref_assembly_csv[ref_assembly_csv['Template_ID']==str(pred_template_id)]


    if len(ref_assembly_filt)==0:
        print(f'No reference document found for doc_id {pred_doc_id}, template {pred_doc_id}, Variation {pred_variation} ...')
    else:
        
        ref_doc_id = ref_assembly_filt.iloc[0]['Doc_ID']
        temp_ref_df = ref_assembly_filt[ref_assembly_filt['Doc_ID']==ref_doc_id]


        for ref_file_name in temp_ref_df['doc_name'].values:
            with open(f'{REF_FOLDER}/{ref_file_name}') as f:
                temp_json = (json.load(f))
            ref_doc_list.append(temp_json)

        if len(ref_doc_list) == 0:
            print(f'No labelled document found for doc_id {pred_doc_id}, template {pred_doc_id}, Variation {pred_variation} ...')

        get_page_labels(pred_doc_id_without_ext, pred_ocr, s3_client, ref_doc_list)

        sync_commnad = f'aws s3 sync "{LOCAL_FINAL_LABELS}" "{FINAL_LABELS_TARGET_S3_PATH}" --quiet'
        os.system(sync_commnad)

        

