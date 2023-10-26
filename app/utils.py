import os
import re
import boto3
import numpy as np
import pandas as pd
from uuid import uuid4
import botocore.exceptions
from rapidfuzz import fuzz
from global_variables import entities_to_label

entities_to_idx = {entities_to_label[idx]:idx for idx in range(len(entities_to_label))}


def get_file_name_from_path(path):
    return path.rsplit('/')[-1].rsplit('.')[0]


def get_cosine_score(vec_a, vec_b):
    score = np.dot(vec_a,vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b) + 1e-8)
    return score


def sync_label_s3_folder(source,target):
    sync_command = f'aws s3 sync {source} {target} --quiet'
    os.system(sync_command)


def create_task_per_image(img_path):
    """
    creates a label-studio task for labelling
    :param img_path: str, s3 image path
    :return: dict, label-studio task
    """
    task_json = {
        "data": {
            "image": img_path
        },
        "predictions": [
            {
                "result": [
                ]
            }
        ]
    }
    return task_json


def download_from_s3(document_path, filename, attachment_bucket, s3_client=None):
    fetch_successful = True
    if not s3_client:
        s3_client = boto3.client('s3')
    try:
        s3_client.download_file(attachment_bucket, document_path, filename)
    except botocore.exceptions.ClientError:
        fetch_successful = False
    return fetch_successful


def download_vision_response(doc_id_without_extension, s3_client, ref=False):
    ocr_output_filename = f'/tmp/{doc_id_without_extension}.parquet'
    ocr_bucket = os.environ['DATA_BUCKET']
    # if ref:
    #     ocr_output_filename_s3 = f'ocr_output_excel/{doc_id_without_extension}.parquet'
    # else:
    #     ocr_output_filename_s3 = f'ocr_output/{doc_id_without_extension}.parquet'
    ocr_output_filename_s3 = f'ocr_output/{doc_id_without_extension}.parquet'
    download_from_s3(ocr_output_filename_s3, ocr_output_filename, ocr_bucket, s3_client)
    word_df = pd.read_parquet(ocr_output_filename)
    return word_df


def get_normalised_bbox_from_label_studio_bbox(value_obj, width=100.0, height=100.0):
    
    return {
        'minx': value_obj['x']/width,
        'miny': value_obj['y']/height,
        'maxx': (value_obj['x']+value_obj['width'])/width,
        'maxy': (value_obj['y']+value_obj['height'])/height
    }


def get_label_studio_bbox_from_normalised_bbox(normalised_bbox, w=100, h=100):
    
    label_studio_bbox = {
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
    }
    
    if not normalised_bbox:
        return label_studio_bbox
    
    label_studio_bbox.update({
        'x': w*normalised_bbox['minx'],
        'y': h*normalised_bbox['miny'],
        'width': w*(normalised_bbox['maxx']-normalised_bbox['minx']),
        'height': h*(normalised_bbox['maxy']-normalised_bbox['miny'])
    })
    
    return label_studio_bbox


def normalise_bbox(bbox, w, h):
    return {
        'minx': bbox['minx']/w,
        'maxx': bbox['maxx']/w,
        'miny': bbox['miny']/h,
        'maxy': bbox['maxy']/h
    }


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    x2 = min(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max((x2 - x1, 0)) * max((y2 - y1), 0)
    if inter_area == 0:
        return 0
    
    box1_area = abs((box1[1] - box1[0]) * (box1[3] - box1[2]))
    box2_area = abs((box2[1] - box2[0]) * (box2[3] - box2[2]))
    # iou = inter_area / float(box1_area + box2_area - inter_area)
    iou = inter_area / min(float(box1_area), float(box2_area))

    return iou


def create_label_studio_object(bbox, label_name, original_width, 
                                                   original_height, text):
    label_studio_obj = {
        "original_width": original_width,
        "original_height": original_height,
        "image_rotation": 0,
        "value": {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "rotation": 0,
            "rectanglelabels": [
                label_name
            ],
            "text": text
        },
        "id": str(uuid4()),
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
    }
    if False in [bbox[key]==bbox[key] for key in bbox.keys()]:
        return label_studio_obj
    
    label_studio_obj['value'].update(**bbox)

    return label_studio_obj


def get_page_match_scores(pred_docs,ref_doc):
    '''
    pred_doc_page : auto labelling script output json document for a page to be lablled in document to be labelled
    ref_doc: list of json data for each page in correctly labelled reference document

    returns: array of page matching scores
    '''
    match_scores = []

    labels_in_ref_page = ref_doc['result']

    ref_page_ent_vec = np.zeros(len(entities_to_label))

    for label in labels_in_ref_page:
        if 'value' in label.keys():
            if 'rectanglelabels' in label['value'].keys():
                if label['value']['rectanglelabels'][0] in entities_to_label:
                    ref_page_ent_vec[entities_to_idx[label['value']['rectanglelabels'][0]]] += 1
    

    for idx in range(len(pred_docs)):
        page_data = pred_docs[idx]
        labels_in_pred = page_data['predictions'][0]['result']
        
        pred_ent_vec = np.zeros(len(entities_to_label))

        for label in labels_in_pred:
            if 'value' in label.keys():
                if 'rectanglelabels' in label['value'].keys():
                    if label['value']['rectanglelabels'][0] in entities_to_label:
                        pred_ent_vec[entities_to_idx[label['value']['rectanglelabels'][0]]] += 1

        cosine_score = get_cosine_score(pred_ent_vec, ref_page_ent_vec)
        match_scores.append(cosine_score)

    # for idx in range(len(ref_doc)):
    #     page_data = ref_doc[idx]
    #     labels_in_ref_page = page_data['result']

    #     ref_page_ent_vec = np.zeros(len(entities_to_label))

    #     for label in labels_in_ref_page:
    #         if 'value' in label.keys():
    #             if 'rectanglelabels' in label['value'].keys():
    #                 if label['value']['rectanglelabels'][0] in entities_to_label:
    #                     ref_page_ent_vec[entities_to_idx[label['value']['rectanglelabels'][0]]] += 1

    #     cosine_score = get_cosine_score(pred_ent_vec, ref_page_ent_vec)
    #     match_scores.append(cosine_score)

    return match_scores


def is_word_special_char_string(word):
    pattern = r'^[^a-zA-Z0-9]+$'
    return bool(re.match(pattern, word))


def filter_special_char_words(word_df):
    cond = word_df['word'].apply(is_word_special_char_string) == False
    cond2 = word_df['word'].apply(lambda word: "------------------" not in word) # for 20236623433886661_2
    word_df = word_df[cond]
    return word_df[cond2]


def get_unnormalised_bbox(normalised_bbox, page_w, page_h):
    return {
        'minx': normalised_bbox['minx']*page_w,
        'maxx': normalised_bbox['maxx']*page_w,
        'miny': normalised_bbox['miny']*page_h,
        'maxy': normalised_bbox['maxy']*page_h
    }


def get_words_within_bbox(bbox, ocr_data):
    temp_word_df = pd.DataFrame(columns=(ocr_data.columns))
    for index, row in ocr_data.iterrows():
        word_bbox = (row['minx'], row['maxx'], row['miny'], row['maxy'])
        temp_bbox = (bbox['minx'], bbox['maxx'], bbox['miny'], bbox['maxy'])
        iou = bbox_iou(temp_bbox,word_bbox)
        if iou>0.3:
            temp_word_df.loc[len(temp_word_df)] = row

    temp_word_df = temp_word_df.sort_values(by=['miny','minx'])
    return temp_word_df, ' '.join(temp_word_df['word'].values)


def expand_bbox(bbox, ratio=2):
    width = bbox['maxx'] - bbox['minx']
    height = bbox['maxy'] - bbox['miny']
    midx  = (bbox['minx']+bbox['maxx'])/2
    midy = (bbox['miny']+bbox['maxy'])/2

    new_width = width*ratio
    new_height = height*ratio

    new_minx  = midx - (new_width/2)
    new_maxx  = midx + (new_width/2)
    new_miny  = midy - (new_height/2)
    new_maxy  = midy + (new_height/2)

    return {
        'minx': new_minx,
        'maxx': new_maxx,
        'miny': new_miny,
        'maxy': new_maxy
    }


def get_bbox_from_df(df):
    return {
        'minx': min(df['minx'].values),
        'maxx': max(df['maxx'].values),
        'miny': min(df['miny'].values),
        'maxy': max(df['maxy'].values)
    }


def clean_item(item):
    output = []
    for s in item:
        if s.isalnum() or s == ' ':
            output.append(s)
    return ''.join(output)


def is_fuzzy_match(data_1, data_2, text_same_threshold = 90):
    is_match = fuzz.ratio(clean_item(str(data_1).strip().lower()), clean_item(str(data_2).strip().lower())) >= text_same_threshold
    return is_match


def is_label_duplicate(label, label_list):
    for ref_label in label_list:
        bbox_pred = get_normalised_bbox_from_label_studio_bbox(label['value'])
        bbox_ref = get_normalised_bbox_from_label_studio_bbox(ref_label['value'])
        pred_bbox = (bbox_pred['minx'], bbox_pred['maxx'], bbox_pred['miny'], bbox_pred['maxy'])
        ref_bbox = (bbox_ref['minx'], bbox_ref['maxx'], bbox_ref['miny'], bbox_ref['maxy'])
        if bbox_iou(pred_bbox,ref_bbox)>0.8:
            if ref_label['value']['rectanglelabels']==label['value']['rectanglelabels']:
                return True
    return False


def find_key_through_relation(value_label_name, ref_page_json):
    value_id = ''
    for label in ref_page_json['result']:
        if label['type'] == 'rectanglelabels' and 'value' in label.keys():
            if 'rectanglelabels' in label['value'] and label['value']['rectanglelabels'][0]==value_label_name:
                value_id = label['id']

    if value_id == '':
        return ''
    
    from_id = ''

    for label in ref_page_json['result']:
        if label['type'] == 'relation':
            if label['to_id'] == value_id:
                from_id = label['from_id']
                break
            
    for label in ref_page_json['result']:
        if label['type'] == 'rectanglelabels':
            if label['id'] == from_id:
                key_name = label['value']['rectanglelabels'][0]
                return key_name
            
    return ''


def get_value_bbox(value_name, key_name, key_bbox, ref_page_json, ocr_data_pred):
    labels_in_ref_page = ref_page_json['result']
    ref_value_bbox = None
    ref_key_bbox = None
    ref_w = None
    ref_h = None
    
    for label in labels_in_ref_page:
            if 'original_width' in label.keys():
                ref_w = label['original_width']
                ref_h = label['original_height']
            if 'value' in label.keys() and 'rectanglelabels' in label['value'].keys() and label['value']['rectanglelabels'][0] == value_name:
                ref_value_bbox = get_normalised_bbox_from_label_studio_bbox(label['value'])
            if 'value' in label.keys() and 'rectanglelabels' in label['value'].keys() and label['value']['rectanglelabels'][0] == key_name:
                ref_key_bbox = get_normalised_bbox_from_label_studio_bbox(label['value'])

    # value_unnormalised_bbox_ref = get_unnormalised_bbox(ref_value_bbox, ref_w, ref_h)
    # key_unnormalised_bbox_ref = get_unnormalised_bbox(ref_key_bbox, ref_w, ref_h)
    value_unnormalised_bbox_ref = ref_value_bbox
    key_unnormalised_bbox_ref = ref_key_bbox

    key_midx = (key_bbox['minx']+key_bbox['maxx'])/2
    key_midy = (key_bbox['miny']+key_bbox['maxy'])/2
    
    ref_key_midx = (key_unnormalised_bbox_ref['minx']+key_unnormalised_bbox_ref['maxx'])/2
    ref_key_midy = (key_unnormalised_bbox_ref['miny']+key_unnormalised_bbox_ref['maxy'])/2

    ref_value_midx = (value_unnormalised_bbox_ref['minx']+value_unnormalised_bbox_ref['maxx'])/2
    ref_value_midy = (value_unnormalised_bbox_ref['miny']+value_unnormalised_bbox_ref['maxy'])/2

    delta_x = ref_value_midx - ref_key_midx
    delta_y = ref_value_midy - ref_key_midy

    value_midx = key_midx + delta_x
    value_midy = key_midy + delta_y

    value_bbox_width = value_unnormalised_bbox_ref['maxx'] - value_unnormalised_bbox_ref['minx']
    value_bbox_height = value_unnormalised_bbox_ref['maxy'] - value_unnormalised_bbox_ref['miny']

    temp_value_bbox = {
        'minx': value_midx - (value_bbox_width/2),
        'maxx': value_midx + (value_bbox_width/2),
        'miny': value_midy - (value_bbox_height/2),
        'maxy': value_midy + (value_bbox_height/2)
    }

    value_df, value_text = get_words_within_bbox(temp_value_bbox, ocr_data_pred)
    if len(value_df)>0:
        value_bbox = get_bbox_from_df(value_df)
        if value_bbox['minx'] > temp_value_bbox['minx']:
            x_shift = value_bbox['minx'] - temp_value_bbox['minx']
            temp_value_bbox['minx'] = temp_value_bbox['minx'] + x_shift
            temp_value_bbox['maxx'] = temp_value_bbox['maxx'] + x_shift
            value_df, value_text = get_words_within_bbox(temp_value_bbox, ocr_data_pred)
            if len(value_df)>0:
                value_bbox = get_bbox_from_df(value_df)
            else:
                value_bbox = {'minx':0, 'maxx':0, 'miny':0, 'maxy':0}
                return value_bbox, value_text

        return value_bbox, value_text
    
    else:
        value_bbox = {'minx':0, 'maxx':0, 'miny':0, 'maxy':0}
        return value_bbox, value_text


def find_key_bbox(key_name, pred_page_json, ref_page_json, ocr_data_pred, ocr_data_ref):
    labels_in_ref_page = ref_page_json['result']
    ref_key_bbox = None
    ref_w = None
    ref_h = None
    
    for label in labels_in_ref_page:
            if 'original_width' in label.keys():
                ref_w = label['original_width']
                ref_h = label['original_height']
            if 'value' in label.keys() and 'rectanglelabels' in label['value'].keys() and label['value']['rectanglelabels'][0] == key_name:
                ref_key_bbox = get_normalised_bbox_from_label_studio_bbox(label['value'])

    labels_in_pred = pred_page_json['predictions'][0]['result']
    pred_w = None
    pred_h = None

    for label in labels_in_pred:
        if 'original_width' in label.keys():
                pred_w = label['original_width']
                pred_h = label['original_height']

    if ref_key_bbox is None:
        return False, '', {}

    # key_unnormalised_bbox_pred = get_unnormalised_bbox(ref_key_bbox, pred_w, pred_h)
    key_unnormalised_bbox_pred = ref_key_bbox
    pred_key_df, pred_key_string = get_words_within_bbox(key_unnormalised_bbox_pred, ocr_data_pred)

    # key_unnormalised_bbox_ref = get_unnormalised_bbox(ref_key_bbox, ref_w, ref_h)
    key_unnormalised_bbox_ref = ref_key_bbox
    _, ref_key_string = get_words_within_bbox(key_unnormalised_bbox_ref, ocr_data_ref)


    if is_fuzzy_match(pred_key_string,ref_key_string) and pred_key_string != '':
        key_bbox  = get_bbox_from_df(pred_key_df)
        return True, pred_key_string, key_bbox
    

    value_name = key_name.replace('_key_name','_key')
    pred_value_bbox = None

    for label in labels_in_pred:
        if label['value']['rectanglelabels'][0] == value_name:
            pred_value_bbox = get_normalised_bbox_from_label_studio_bbox(label['value'])

    if pred_value_bbox is not None:
        
        # pred_value_bbox = get_unnormalised_bbox(pred_value_bbox, pred_w, pred_h)

        pred_value_bbox = expand_bbox(pred_value_bbox)
        expanded_word_df , words_around_value = get_words_within_bbox(pred_value_bbox, ocr_data_pred)

        temp_df = pd.DataFrame(columns=(ocr_data_pred.columns))

        for word in ref_key_string.split(' '):
            if word in expanded_word_df['word'].values:
                temp_df.loc[len(temp_df)] = expanded_word_df[expanded_word_df['word']==word].reset_index(drop=True).iloc[0]
        
        if len(temp_df)!=0:
            key_bbox = get_bbox_from_df(temp_df)
            _ , bbox_word_string = get_words_within_bbox(key_bbox, ocr_data_pred)
            if is_fuzzy_match(bbox_word_string,ref_key_string):
                return True, ''.join(temp_df['word'].values), key_bbox

    # 3rd Fallback - entire OCR Search
    temp_df = pd.DataFrame(columns=(ocr_data_pred.columns))

    word_to_df_dict = {}

    for word in ref_key_string.split(' '):
        temp_word_df = pd.DataFrame(columns=(ocr_data_pred.columns))
        if word in ocr_data_pred['word'].values:
            if len(temp_word_df)==0:
                temp_word_df = ocr_data_pred[ocr_data_pred['word']==word].reset_index(drop=True)
            else:
                temp_word_df = pd.concat([temp_word_df,ocr_data_pred[ocr_data_pred['word']==word].reset_index(drop=True)])
            word_to_df_dict[word]=temp_word_df

    all_df_list = get_all_possible_comibnations([temp_df],list(word_to_df_dict.values()))

    match_list = []
    for temp_df in all_df_list:
        if len(temp_df)!=0:
            key_bbox = get_bbox_from_df(temp_df)
            _ , bbox_word_string = get_words_within_bbox(key_bbox, ocr_data_pred)
            if bbox_word_string == ref_key_string:
                match_list.append((True, ' '.join(temp_df['word'].values), key_bbox))
    if len(match_list)==1:
        return match_list[0][0], match_list[0][1], match_list[0][2]
    
    return False, '', {}


def find_header_bbox(key_name, pred_page_json, ref_page_json, ocr_data_pred, ocr_data_ref):
    labels_in_ref_page = ref_page_json['result']
    ref_key_bbox = None
    ref_w = None
    ref_h = None
    
    for label in labels_in_ref_page:
            if 'original_width' in label.keys():
                ref_w = label['original_width']
                ref_h = label['original_height']
            if 'value' in label.keys() and 'rectanglelabels' in label['value'].keys() and label['value']['rectanglelabels'][0] == key_name:
                ref_key_bbox = get_normalised_bbox_from_label_studio_bbox(label['value'])

    labels_in_pred = pred_page_json['predictions'][0]['result']
    pred_w = None
    pred_h = None

    for label in labels_in_pred:
        if 'original_width' in label.keys():
                pred_w = label['original_width']
                pred_h = label['original_height']

    if ref_key_bbox is None:
        return False, '', {}

    # key_unnormalised_bbox_pred = get_unnormalised_bbox(ref_key_bbox, pred_w, pred_h)
    key_unnormalised_bbox_pred = ref_key_bbox
    pred_key_df, pred_key_string = get_words_within_bbox(key_unnormalised_bbox_pred, ocr_data_pred)

    # key_unnormalised_bbox_ref = get_unnormalised_bbox(ref_key_bbox, ref_w, ref_h)
    key_unnormalised_bbox_ref = ref_key_bbox
    _, ref_key_string = get_words_within_bbox(key_unnormalised_bbox_ref, ocr_data_ref)


    if is_fuzzy_match(pred_key_string,ref_key_string) and len(pred_key_df):
        key_bbox = get_bbox_from_df(pred_key_df)
        return True, pred_key_string, key_bbox
    

    # 2nd Fallback for Header search- entire OCR Search
    temp_df = pd.DataFrame(columns=(ocr_data_pred.columns))

    word_to_df_dict = {}

    for word in ref_key_string.split(' '):
        temp_word_df = pd.DataFrame(columns=(ocr_data_pred.columns))
        if word in ocr_data_pred['word'].values:
            if len(temp_word_df)==0:
                temp_word_df = ocr_data_pred[ocr_data_pred['word']==word].reset_index(drop=True)
            else:
                temp_word_df = pd.concat([temp_word_df,ocr_data_pred[ocr_data_pred['word']==word].reset_index(drop=True)])
            word_to_df_dict[word]=temp_word_df

    all_df_list = get_all_possible_comibnations([temp_df],list(word_to_df_dict.values()))

    match_list = []
    for temp_df in all_df_list:
        if len(temp_df)!=0:
            key_bbox = get_bbox_from_df(temp_df)
            _ , bbox_word_string = get_words_within_bbox(key_bbox, ocr_data_pred)
            if bbox_word_string == ref_key_string:
                match_list.append((True, ' '.join(temp_df['word'].values), key_bbox))
    if len(match_list)>=1:
        return match_list[0][0], match_list[0][1], match_list[0][2]
    
    return False, '', {}


def get_all_possible_comibnations(return_df_list, df_list):
    if len(df_list)==0:
        return return_df_list
    elif len(df_list)==1:
        new_return_list = []
        for i in range(len(df_list[0])):
            for df in return_df_list:
                new_df = df.copy()
                new_df.loc[len(new_df)] = df_list[0].iloc[i]
                new_return_list.append(new_df)

        return new_return_list
    else:
        new_return_list = []
        for i in range(len(df_list[0])):
            for df in return_df_list:
                new_df = df.copy()
                new_df.loc[len(new_df)] = df_list[0].iloc[i]
                new_return_list.append(new_df)
        return get_all_possible_comibnations(new_return_list, df_list[1:])
        

def remove_headers(page_labels):
    new_labels = []
    for label in page_labels:
        if 'value' in label.keys():
            if 'rectanglelabels' in label['value']:
                label_name = label['value']['rectanglelabels'][0]
                if not '_header' in label_name:
                    new_labels.append(label)

    return new_labels 


def is_headers_span_large(page_labels):
    miny = 1000
    maxy = 0
    for label in page_labels:
        if 'value' in label.keys():
            if 'rectanglelabels' in label['value']:
                label_name = label['value']['rectanglelabels'][0]
                if '_header' in label_name:
                    if label['value']['y'] < miny:
                        miny = label['value']['y']
                    if label['value']['y'] + label['value']['height'] > maxy:
                        maxy = label['value']['y'] + label['value']['height']
    if maxy-miny>10:
        return True   
    else:
        return False
    


def get_col_iou(header_label,cell_label):
    x1, w1 = header_label['value']['x'], header_label['value']['width']
    x2, w2 = cell_label['value']['x'], cell_label['value']['width']

    rx = min(x1+w1,x2+w2)
    lx = max(x1,x2)

    if rx>lx:
        iou = (rx-lx)/(min(w1,w2)+1e-6)
    else:
        iou = 0

    return iou
    

def change_table_labels(page_labels, header_page = []):
    if not header_page:
        header_page = page_labels
    final_labels = []
    header_labels = {}
    for label in header_page:
        if 'value' in label.keys():
            if 'rectanglelabels' in label['value']:
                label_name = label['value']['rectanglelabels'][0]
                if label_name.endswith('_header'):
                    header_labels[label_name] = label
    
    if len(header_labels.keys())==0:
        return page_labels
    
    for label in page_labels:
        if 'value' in label.keys():
            if 'rectanglelabels' in label['value']:
                label_name = label['value']['rectanglelabels'][0]
                if label_name.endswith('_table'):
                    header_name = label_name.replace('_table','_header')
                    if header_name in header_labels.keys() and get_col_iou(header_labels[header_name],label)>0:
                        final_labels.append(label)
                    else:
                        iou_scores = []
                        header_names = []
                        for header_name,header_label in header_labels.items():
                            iou_scores.append(get_col_iou(header_label,label))
                            header_names.append(header_name)
                        max_idx = iou_scores.index(max(iou_scores)) if max(iou_scores)!=0 else -1
                        if max_idx == -1 or len([score for score in iou_scores if score==max(iou_scores)])>1:
                            final_labels.append(label)
                            continue
                        header_name = header_names[max_idx]
                        label['value']['rectanglelabels'][0] = header_name.replace('_header','_table')
                        final_labels.append(label)
                else:
                    final_labels.append(label)
            else:
                final_labels.append(label)
        else:
            final_labels.append(label)

    return final_labels

                        