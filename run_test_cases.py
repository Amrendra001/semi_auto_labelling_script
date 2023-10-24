import sys
sys.path.append('app/')
from dotenv import load_dotenv
load_dotenv()
from app import lambda_handler


if __name__ == '__main__':
   
    event = {
        "doc_id": "202321545422673202.XLS",
        "template_id": "7122",
        "key_acc_id": "1011"
    }

    lambda_handler(event, '')
