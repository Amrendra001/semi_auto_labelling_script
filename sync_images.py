import os

doc_id = '202242064856644882'
for i in range(3):
    img_name = f'{doc_id}_{i}.png'
    path_1 = f's3://document-ai-training-data/common_format/1698079980/images/{img_name}'
    path_2 = f's3://document-ai-training-data/common_format/1698057359/images/{img_name}'
    os.system(f'aws s3 cp {path_1} {path_2}')