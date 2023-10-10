import requests
import hashlib
import time
import random

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

def init_client():
    res = get_auth()

    secret_id = res["TmpSecretId"]
    secret_key = res["TmpSecretKey"]
    region = 'ap-shanghai'
    token = res["Token"]

    config = CosConfig(
        Region=region,
        SecretId=secret_id,
        SecretKey=secret_key,
        Token=token,
    )

    return CosS3Client(config)

def get_auth():
    return requests.get(f'http://api.weixin.qq.com/_/cos/getauth').json()

def get_file_by_id(cloud_path):
    client = init_client()
    rsp = client.get_object(
        Bucket='7072-prod-1gm7qkcd9e563f22-1321279255',
        Key=cloud_path,
    )
    
    return rsp["Body"].get_raw_stream()

def upload_file(file_path):
    md5_obj = hashlib.md5()
    md5_obj.update(f'{time.time() * 1000}_{random.random() * 1000}'.encode())
    client = init_client()
    file_name = f'detect_image_output/{md5_obj.hexdigest()}'
    client.put_object_from_local_file(
        Bucket='7072-prod-1gm7qkcd9e563f22-1321279255',
        LocalFilePath=file_path,
        Key=file_name,
    )
    return f'cloud://prod-1gm7qkcd9e563f22.7072-prod-1gm7qkcd9e563f22-1321279255/{file_name}'