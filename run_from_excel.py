import os
import json
from tqdm import tqdm
import pandas as pd
import boto3

# DATASET_PATH = '/Javis/Testing_Lambda/ai-central-testing-repo/information_extraction/data/dataset'
QUEUE_URL = 'https://sqs.ap-south-1.amazonaws.com/144366017634/semi_auto_labelling_queue'


def add_message_to_sqs(event, sqs_client=None):
    if not sqs_client:
        sqs_client = boto3.client('sqs')
    response = sqs_client.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(event),
        DelaySeconds=0)
    # print('Message Sent:', response['MessageId'])


def main():
    sqs_client = boto3.client('sqs')

    # dataset = datasets.load_from_disk(f'{DATASET_PATH}', keep_in_memory=False)

    # doc_ids_list = []
    # for i, example in enumerate(dataset['test']):
    #     doc_ids_list.append(example['doc_id'])
    # for i, example in enumerate(dataset['val']):
    #     doc_ids_list.append(example['doc_id'])
    # for i, example in enumerate(dataset['train']):
    #     doc_ids_list.append(example['doc_id'])
    
    MT_docs_df = pd.read_csv('app/MT_merged_big_excel_set_with_variations.csv')
    # doc_ids_list = MT_docs_df['Doc_ID'].values

    # doc_ids_list = list(set(doc_ids_list))

    print('Sending messages to the SQS...')

    # MT_assembly_line_df = pd.read_csv('mt_migration_20220401_20230420_assembly_line_new.csv')
    count = 0

    for i in tqdm(range(len(MT_docs_df))):
        row_data = MT_docs_df.iloc[i]
        doc_id = row_data['Doc_ID']
        template_id = str(row_data['Template_ID'])
        # if template_id != '161':
        #     continue
        key_acc_id = str(row_data['Javis_DB_ID'])
        event = {
            'doc_id': doc_id,
            'template_id': template_id,
            'key_acc_id': key_acc_id
        }
        add_message_to_sqs(event, sqs_client)
        count += 1

    print(f'Done. Sent {count} messages to the SQS')


if __name__ == '__main__':
    main()



