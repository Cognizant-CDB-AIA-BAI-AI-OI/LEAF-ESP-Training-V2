import click
import os
import tempfile
from uuid import uuid4
import boto3
import json

from botocore.exceptions import ClientError


class S3Util(object):
    """
    A class that handles the interactions with S3 and the file system
    """
    s3 = None

    def __init__(self, aws_access_id=None, aws_secret_key=None):
        """
        Create a new S3Util class
        :param aws_access_id: the aws account access id
        :param aws_secret_key: the aws account secret key
        """
        if aws_access_id and aws_secret_key:
            self.s3 = boto3.resource('s3',
                                     aws_access_key_id=aws_access_id,
                                     aws_secret_access_key=aws_secret_key,
                                     region_name='us-west-2')
        else:
            # default credentials/IAM scenario
            self.s3 = boto3.resource('s3')

    def list_objects(self, bucket_name, s3_folder_key):
        """
        Obtains a list of objects from S3 from a specified bucket and prefix
        :param bucket_name: The bucket to be queried in S3
        :param s3_folder_key: The prefix to be queried. Only objects beginning with this prefix will be returned.
        :return: A list of objects matching the query.
        """
        bucket = self.s3.Bucket(bucket_name)
        return bucket.objects.filter(Prefix=s3_folder_key)

    def upload_model(self, model_bucket_name, s3_folder_key, model):
        """
        Uploads the passed model to s3

        :param model_bucket_name: the name of the S3 bucket containing the
        models
        :param s3_folder_key: the name of the folder in the bucket containing
        the models
        :param model: the model to upload
        :return: the generated unique uuid.hd5 file name that contains the
        model in the S3 bucket/s3_folder_key.
        """
        # The model serializer will close the file, so use delete=False
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            model.save(temp_file.name)
            bucket = self.s3.Bucket(model_bucket_name)
            with open(temp_file.name, 'rb') as saved_model:
                model_file_name = '{}.hd5'.format(uuid4())
                bucket.upload_fileobj(saved_model, '{}/{}'.format(
                    s3_folder_key, model_file_name))
        finally:
            os.remove(temp_file.name)
        return model_file_name

    def download_file(self, model_bucket_name, s3_key, file_name):
        """
        Downloads a file from s3
        :param model_bucket_name: the name of the S3 bucket
        :param s3_key = the 'folder' containing the file
        :param file_name = the name of the file
        :return: the local path to which the model file was temporarily
        downloaded
        """
        # File path
        s3_path = '{}/{}'.format(s3_key, file_name)
        # Bucket
        try:
            bucket = self.s3.Bucket(model_bucket_name)
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            bucket.download_file(s3_path, temp_file.name)
            return temp_file.name
        except ClientError:
            raise IOError('Unable to retrieve file {file} from S3'.format(file=s3_path))

    @staticmethod
    def remove_temp_file(temp_file):
        """
        Removes the file corresponding to the passed file name
        :param temp_file: a file name string
        :return: nothing
        """
        if temp_file:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def upload_dictionary(self, bucket_name, file_name, dictionary):
        """
        Uploads the passed json string to s3

        :param bucket_name: the name of the S3 bucket to upload to
        :param file_name: the full name of the file to upload in the bucket
        :param dictionary: the dictionary to upload as a file
        :return: nothing
        """
        s3_object = self.s3.Object(bucket_name, file_name)
        s3_object.put(Body=json.dumps(dictionary))

    def download_dictionary(self, bucket_name, file_name):
        """
        Downloads and returns the passed file name as a dictionary
        :param bucket_name: the name of the S3 bucket to download from
        :param file_name: the full name of a json file
        :return: a dictionary
        """
        s3_object = self.s3.Object(bucket_name, file_name)
        data = s3_object.get()['Body'].read()
        return json.loads(data)


@click.group()
@click.option('--aws_access_id', envvar='AWS_ACCESS_ID')
@click.option('--aws_secret_key', envvar='AWS_SECRET_KEY')
@click.pass_context
def _cli(ctx, aws_access_id, aws_secret_key):
    ctx.obj['S3_CLIENT'] = S3Util(aws_access_id, aws_secret_key)


@_cli.command()
@click.option('--model_bucket_name')
@click.option('--s3_folder_key')
@click.option('--model')
@click.pass_context
def upload_model(ctx, model_bucket_name, s3_folder_key, model):
    """
    Uploads a model to S3.
    :param ctx: the click context containing the S3 client created using --aws_access_id and --aws_secret_key
    :param model_bucket_name: the name of the S3 bucket
    :param s3_folder_key: the folder in the S3 bucket
    :param model: the model's .hd5 file to upload
    :return: the name of the file created in S3
    """
    s3_client = ctx.obj['S3_CLIENT']
    bucket = s3_client.s3.Bucket(model_bucket_name)
    with open(model, 'rb') as saved_model:
        model_file_name = '{}.hd5'.format(uuid4())
        bucket.upload_fileobj(saved_model, '{}/{}'.format(
            s3_folder_key, model_file_name))
        print("File successfully uploaded. Name is {}".format(model_file_name))


@_cli.command()
@click.option('--model_bucket_name')
@click.option('--s3_folder_key')
@click.option('--model_file_name')
@click.pass_context
def download_model(ctx, model_bucket_name, s3_folder_key, model_file_name):
    """
    Downloads a model from S3.
    :param ctx: the click context containing the S3 client created using --aws_access_id and --aws_secret_key
    :param model_bucket_name: the name of the S3 bucket
    :param s3_folder_key: the folder in the S3 bucket
    :param model_file_name: name of the model's .hd5 file to download
    :return: the name of the file created locally
    """
    s3_client = ctx.obj['S3_CLIENT']
    # File path
    s3_path = '{}/{}'.format(s3_folder_key, model_file_name)
    # Bucket
    bucket = s3_client.s3.Bucket(model_bucket_name)
    bucket.download_file(s3_path, model_file_name)
    print("File successfully downloaded: {}".format(model_file_name))


if __name__ == '__main__':
    _cli(obj={})
