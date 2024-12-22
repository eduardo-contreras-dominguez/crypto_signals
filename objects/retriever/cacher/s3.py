import boto3
from loguru import logger
import io
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from dotenv import load_dotenv
import os
load_dotenv()
class S3:
    @staticmethod
    def get_s3_client():

        aws_access_key_id =  os.getenv('AWS_ACCESS_KEY')
        aws_secret_access_key = os.getenv('AWS_SECRET_KEY')

        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    @staticmethod
    def upload_parquet_to_s3(dataframe, bucket_name, object_key):

        table = pa.Table.from_pandas(dataframe)
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        s3_client = S3.get_s3_client()
        s3_client.upload_fileobj(buffer, bucket_name, object_key)
        logger.success(f"Parquet correctly uploaded to s3://{bucket_name}/{object_key}")

    @staticmethod
    def read_parquet_from_s3(bucket_name, object_key):
        s3_client = S3.get_s3_client()
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket_name, object_key, buffer)
        buffer.seek(0)

        # Leer el Parquet en un dataframe de pandas
        table = pq.read_table(buffer)
        logger.success('Parquet file read correctly')
        return table.to_pandas()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear un dataframe de ejemplo
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })

    # Especificar el nombre del bucket y la clave del objeto
    bucket = 'cryptoprices'
    object_key = 'TEST.parquet'

    # Subir el dataframe como Parquet a S3
    S3.upload_parquet_to_s3(df, bucket, object_key)

    # Leer el archivo Parquet desde S3
    df_leido = S3.read_parquet_from_s3(bucket, object_key)
    print(df_leido)