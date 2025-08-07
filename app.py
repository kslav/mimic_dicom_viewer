# app.py

import os
import io
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import boto3
import pydicom
from botocore.exceptions import ClientError

# ——— ❶ Configuration ———
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EXPIRATION = 3600  # presigned URL valid for 1 hour

# Create an S3 client once
s3 = boto3.client("s3", region_name=AWS_REGION)


# ——— ❷ Helpers ———
def parse_s3_uri(uri: str):
    """
    Given 's3://bucket/path/to/file.dcm', returns ('bucket', 'path/to/file.dcm').
    """
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def generate_presigned_url(bucket: str, key: str) -> str:
    """
    Return a presigned GET URL for the S3 object, or empty string on error.
    """
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=EXPIRATION,
        )
    except ClientError as e:
        st.error(f"Could not generate URL for {bucket}/{key}: {e}")
        return ""


@st.cache_data
def load_mapping(csv_file) -> pd.DataFrame:
    """
    Read the uploaded CSV into a DataFrame. Must have STUDYID and S3_PATH columns.
    """
    return pd.read_csv(csv_file)


# ——— ❸ Streamlit UI ———
def main():
    st.title("MIMIC-CXR DICOM Reviewer")
    st.markdown(
        """
        Upload a CSV mapping STUDYID → S3_PATH (full s3://… URIs).  
        The app will generate download links and render a preview of each DICOM.
        """
    )

    # ❶ Upload
    uploaded = st.file_uploader(
        "Upload CSV with columns: STUDYID, S3_PATH", type="csv"
    )
    if not uploaded:
        st.info("Awaiting your mapping CSV...")
        return

    # ❷ Load
    df = load_mapping(uploaded)
    if "STUDYID" not in df.columns or "S3_PATH" not in df.columns:
        st.error("CSV must contain exactly these columns: STUDYID, S3_PATH")
        return

    st.write(f"Loaded {len(df)} rows. Preview:")
    st.dataframe(df.head())

    # ❸ Iterate by STUDYID
    for sid, group in df.groupby("STUDYID"):
        st.header(f"Study ID: {sid}")

        # If multiple rows per STUDYID, loop through them
        for idx, row in group.iterrows():
            uri = row["S3_PATH"]
            bucket, key = parse_s3_uri(uri)

            # ❹ Presigned download link
            url = generate_presigned_url(bucket, key)
            if url:
                st.markdown(f"- [Download raw DICOM]({url})")
            else:
                st.warning(f"- Could not generate link for {uri}")
                continue

            # ❺ Preview in-stream
            obj = s3.get_object(Bucket=bucket, Key=key)
            dicom_bytes = obj["Body"].read()
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
            img = ds.pixel_array

            st.image(img, caption=key.split("/")[-1], use_column_width=True)

    st.success("Done!")

if __name__ == "__main__":
    main()
