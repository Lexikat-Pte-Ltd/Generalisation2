import polars as pl
from io import BytesIO
import s3fs

from typing import cast


bucket = "s3://v1"
destination = "s3://bucket//myfile.parquet"
endpoint = "http://127.0.0.1:8333"
fs = s3fs.S3FileSystem(
    key="your-access-key",  # Access key
    secret="your-secret-key",  # Secret key
    client_kwargs={"endpoint_url": "http://127.0.0.1:8333"},
)

if __name__ == "__main__":
    
    df = pl.DataFrame()

    # write parquet
    with fs.open(destination, mode="wb") as f:
        df.write_parquet(cast(BytesIO, f))
