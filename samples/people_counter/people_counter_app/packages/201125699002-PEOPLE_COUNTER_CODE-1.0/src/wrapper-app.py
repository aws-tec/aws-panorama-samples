import boto3
import os
import sys
import shutil

# Please replace <target_s3_folder> with your own s3 folder
target_s3_folder = "src"
# Please replace <target_s3_bucket> with your own s3 bucket
target_s3_bucket = "panorama-tailgating"
# Please replace <homedir> with the directory you want to download the target
homedir = "/opt/aws/panorama/storage/"

entry_point = "src/app.py"

if os.path.exists(os.path.join(homedir, target_s3_folder)):
    shutil.rmtree(os.path.join(homedir, target_s3_folder))


def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName)
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        if obj.key.endswith('/'): continue
        if not os.path.exists(os.path.join(homedir , os.path.dirname(obj.key))):
            os.makedirs(os.path.join(homedir, os.path.dirname(obj.key)))
        print(os.path.join(homedir , os.path.dirname(obj.key)))
        bucket.download_file(obj.key, os.path.join(homedir, obj.key))

downloadDirectoryFroms3(target_s3_bucket, target_s3_folder)
os.execl(sys.executable, "python3", os.path.join(homedir, entry_point))
