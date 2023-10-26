import os
import json
import boto3
import numpy as np
import pandas as pd
from utils import get_file_name_from_path, download_vision_response, get_page_match_scores, find_key_bbox, \
                    get_value_bbox, get_label_studio_bbox_from_normalised_bbox, normalise_bbox, \
                    create_label_studio_object, find_key_through_relation, create_task_per_image, \
                    is_label_duplicate, find_header_bbox, remove_headers, is_headers_span_large, change_table_labels

TMP_FOLDER = '/tmp/ie_semi_auto_labelling'
REF_FOLDER = '/tmp/ie_semi_auto_labelling/ref'
PRED_FOLDER = '/tmp/ie_semi_auto_labelling/pred'
FINAL_LABEL_FOLDER = '/tmp/ie_semi_auto_labelling/final'

os.makedirs(TMP_FOLDER,exist_ok=True)
os.makedirs(REF_FOLDER,exist_ok=True)
os.makedirs(PRED_FOLDER,exist_ok=True)
os.makedirs(FINAL_LABEL_FOLDER,exist_ok=True)

PRED_DOC_S3_PATH = os.environ['PRED_DOC_S3_PATH']


def check_header_label(page_ls):
    for label_dict in page_ls:
        label = label_dict['value']['rectanglelabels'][0]
        if '_header' in label:
            return True
    return False


def get_page_labels(pred_doc_id, ocr_data_pred_allpage, s3_client = None, ref_json = []):
    # s3_cp_cmd_ref = f'aws s3 cp {REF_DOC_S3_PATH} {REF_FOLDER}/21418.json'
    # os.system(s3_cp_cmd_ref)
    s3_cp_cmd_pred = f'aws s3 sync "{PRED_DOC_S3_PATH}" "{PRED_FOLDER}/{pred_doc_id}" --exclude "*" --include "*{pred_doc_id}_*"'
    os.system(s3_cp_cmd_pred)

    # ref_doc_name_list = ['21383','21384','21385','21386','21387','21388','21389','21390','21391','21392'] RIL
    # ref_doc_name_list = ['21393','21394','21395','21418']
    pred_doc_name_list = os.listdir(f'{PRED_FOLDER}/{pred_doc_id}')

    # ref_json = []
    # for ref_doc_name in ref_doc_name_list:
    #     with open(f'{REF_FOLDER}/{ref_doc_name}.json') as f:
    #         ref_json.append(json.load(f))
    if len(pred_doc_name_list)==0:
        print(f'No initial labels found for doc {pred_doc_id} ...')
        return None

    pred_json = []
    for pred_doc_name in pred_doc_name_list:
        with open(f'{PRED_FOLDER}/{pred_doc_id}/{pred_doc_name}') as f:
            pred_json.append(json.load(f))

    if not s3_client:
        s3_client = boto3.client('s3')

    os.environ['DATA_BUCKET'] = 'javis-ai-parser-dev'

    doc_id_without_ext_ref = '_'.join(get_file_name_from_path(ref_json[0]['task']['data']['image']).split('_')[:-1])
    doc_id_without_ext_pred = '_'.join(get_file_name_from_path(pred_json[0]['data']['image']).split('_')[:-1])

    
    # pred_page = int(get_file_name_from_path(pred_json['data']['image']).split('_')[-1])

    ocr_data_ref_allpage = download_vision_response(doc_id_without_ext_ref, s3_client, ref=True)
    # ocr_data_pred_allpage = download_vision_response(doc_id_without_ext_pred, s3_client)

    pagewise_final_results = {}

    header_keys = set()

    for ref_page_json in ref_json:
        
        page_match_scores = get_page_match_scores(pred_json, ref_page_json)

        # ref_page_json = ref_json[np.argmax(np.asarray(page_match_scores))]

        ref_page =  int(get_file_name_from_path(ref_page_json['task']['data']['image']).split('_')[-1])

        print(f'Processing Page {ref_page} from ref doc {doc_id_without_ext_ref}')

        ref_page_keys = set()


        for label in ref_page_json['result']:
            if 'value' in label.keys():
                if 'rectanglelabels' in label['value']:
                    label_name = label['value']['rectanglelabels'][0]
                    if label_name.endswith('_key'):
                        ref_page_keys.add(label_name)
                    elif label_name.endswith('_header'):
                        header_keys.add(label_name)

        total_keys = len(list(ref_page_keys))

        matched_keys = set()
        
        best_match_ranking = np.argsort(-np.asarray(page_match_scores))

        for idx in range(len(best_match_ranking.tolist())):
            pred_page_idx = best_match_ranking.tolist()[idx]
            pred_page_json = pred_json[pred_page_idx]

            pred_page = int(get_file_name_from_path(pred_page_json['data']['image']).split('_')[-1])

            ocr_data_pred = ocr_data_pred_allpage[ocr_data_pred_allpage['page']==pred_page].reset_index(drop=True)
            ocr_data_ref = ocr_data_ref_allpage[ocr_data_ref_allpage['page']==ref_page].reset_index(drop=True)

            labels_in_pred = pred_page_json['predictions'][0]['result']
            pred_w = None
            pred_h = None

            page_matched_headers = set()

            for label in labels_in_pred:
                if 'original_width' in label.keys():
                        pred_w = label['original_width']
                        pred_h = label['original_height']

            if pred_page_idx in pagewise_final_results.keys():
                final_results = pagewise_final_results[pred_page_idx]
            else:
                final_results = []

            for label_name in list(ref_page_keys.difference(matched_keys).union(header_keys)):
                if label_name.endswith('_key'):
                    key_name = label_name.replace('_key','_key_name')
                    key_found, key_string, key_bbox = find_key_bbox(key_name, pred_page_json, ref_page_json, ocr_data_pred, ocr_data_ref)
                    if key_found:
                        value_bbox, value_text = get_value_bbox(label_name, key_name, key_bbox, ref_page_json, ocr_data_pred)
                        value_bbox = get_label_studio_bbox_from_normalised_bbox(value_bbox)
                        value_label_obj = create_label_studio_object(value_bbox, label_name, pred_w, pred_h, value_text)
                        key_bbox = get_label_studio_bbox_from_normalised_bbox(key_bbox)
                        key_label_obj = create_label_studio_object(key_bbox, key_name, pred_w, pred_h, key_string)
                        if value_text != '' and not is_label_duplicate(value_label_obj, final_results):
                            final_results.append(value_label_obj)
                            matched_keys.add(label_name)
                        if not is_label_duplicate(key_label_obj, final_results):
                            final_results.append(key_label_obj)
                        matched_keys.add(key_name)
                    else:
                        key_name = find_key_through_relation(label_name, ref_page_json)
                        if key_name == '':
                            continue
                        key_found, key_string, key_bbox = find_key_bbox(key_name, pred_page_json, ref_page_json, ocr_data_pred, ocr_data_ref)
                        if key_found:
                            value_bbox, value_text = get_value_bbox(label_name, key_name, key_bbox, ref_page_json, ocr_data_pred)
                            value_bbox = get_label_studio_bbox_from_normalised_bbox(value_bbox)
                            value_label_obj = create_label_studio_object(value_bbox, label_name, pred_w, pred_h, value_text)
                            key_bbox = get_label_studio_bbox_from_normalised_bbox(key_bbox)
                            key_label_obj = create_label_studio_object(key_bbox, key_name, pred_w, pred_h, key_string)
                            matched_keys.add(key_name)
                            if value_text != '' and not is_label_duplicate(value_label_obj, final_results):
                                final_results.append(value_label_obj)
                                matched_keys.add(label_name)
                            if not is_label_duplicate(key_label_obj, final_results):
                                final_results.append(key_label_obj)
                elif '_header' in label_name:
                    header_found, header_string, header_bbox = find_header_bbox(label_name, pred_page_json, ref_page_json, ocr_data_pred, ocr_data_ref)
                    if header_found:
                        header_bbox = get_label_studio_bbox_from_normalised_bbox(header_bbox)
                        header_label_obj = create_label_studio_object(header_bbox, label_name, pred_w, pred_h, header_string)
                        page_matched_headers.add(label_name)
                        if not is_label_duplicate(header_label_obj, final_results):
                            final_results.append(header_label_obj)
                            matched_keys.add(label_name)
    
            if len(list(page_matched_headers))==1:
                final_results = remove_headers(final_results)
            # elif len(list(page_matched_headers))>1 and is_headers_span_large(final_results):
            #     final_results = remove_headers(final_results)
            
            pagewise_final_results[pred_page_idx] = final_results
        
            if len(list(ref_page_keys.difference(matched_keys)))==0:
                if (idx+1)<len(page_match_scores) and page_match_scores[best_match_ranking[idx+1]]==page_match_scores[best_match_ranking[idx]]:
                    matched_keys = set()
                    continue
                elif (idx+1)<len(page_match_scores) and page_match_scores[best_match_ranking[idx]]-page_match_scores[best_match_ranking[idx+1]] < 0.2:
                    matched_keys = set()
                    continue
                else:
                    break

            if (idx+1)<len(page_match_scores) and page_match_scores[best_match_ranking[idx+1]]==page_match_scores[best_match_ranking[idx]]:
                matched_keys = set()
                continue
            elif (idx+1)<len(page_match_scores) and page_match_scores[best_match_ranking[idx]]-page_match_scores[best_match_ranking[idx+1]] < 0.2:
                matched_keys = set()
                continue


    for pred_page_idx in range(len(pred_json)):
        pred_page_json = pred_json[pred_page_idx]
        labels_in_pred = pred_page_json['predictions'][0]['result']
        final_results = []
        for label in labels_in_pred:
            if 'value' in label.keys():
                if 'rectanglelabels' in label['value']:
                    label_name = label['value']['rectanglelabels'][0]
                    if '_key' in label_name or '_header' in label_name:
                        continue
                    else:
                        final_results.append(label)
                else:
                    final_results.append(label)
            else:
                final_results.append(label)

        if sum([check_header_label(page_ls) for i, page_ls in pagewise_final_results.items()]) == 1:
            header_page = [page_ls for i, page_ls in pagewise_final_results.items() if check_header_label(page_ls)][0]
            print('Headers only on one page')
        else:
            header_page = []
            print('Headers on Multi-page')


        if pred_page_idx in pagewise_final_results.keys():
                old_labels = pagewise_final_results[pred_page_idx]
                old_labels.extend(final_results)
                final_results = change_table_labels(old_labels, header_page)
                pagewise_final_results[pred_page_idx] = final_results
        else:
            final_results = change_table_labels(final_results)
            pagewise_final_results[pred_page_idx] = final_results



        task_json = create_task_per_image(pred_page_json['data']['image'])
        task_json['predictions'][0]['result'] = pagewise_final_results[pred_page_idx]

        json_name = get_file_name_from_path(pred_page_json['data']['image'])+'.json'
        json.dump(task_json, open(f'{FINAL_LABEL_FOLDER}/{json_name}', 'w'))


if __name__ == '__main__':
    pred_doc_id_without_ext = '202210114445323232'
    s3_client = boto3.client('s3')
    pred_ocr = download_vision_response(pred_doc_id_without_ext, s3_client)
    get_page_labels(pred_doc_id_without_ext,pred_ocr)



