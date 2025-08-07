# app.py

import os
import io
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import boto3
import pydicom
import os
import streamlit as st
import boto3
from botocore.exceptions import ClientError

# --- Credentials & client bootstrap ---
def _get_aws_creds():
    # Prefer Streamlit Secrets; fall back to env. Strip whitespace just in case.
    ak = (st.secrets.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    sk = (st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    tk = (st.secrets.get("AWS_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN") or "").strip()  # optional
    rg = (st.secrets.get("AWS_REGION") or os.getenv("AWS_REGION") or "us-east-1").strip()
    return ak, sk, tk, rg

AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION = _get_aws_creds()

# Minimal debug (safe/temporary): show presence and shapes, not the secrets.
with st.expander("Debug: AWS creds (safe view)"):
    def mask(x): 
        return f"{x[:4]}...{x[-4:]} (len={len(x)})" if x else "MISSING"
    st.write({
        "AWS_ACCESS_KEY_ID": mask(AWS_ACCESS_KEY_ID),
        "AWS_SECRET_ACCESS_KEY": f"(len={len(AWS_SECRET_ACCESS_KEY)})" if AWS_SECRET_ACCESS_KEY else "MISSING",
        "AWS_SESSION_TOKEN": "present" if AWS_SESSION_TOKEN else "absent",
        "AWS_REGION": AWS_REGION,
    })

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    st.error("AWS credentials not found. Check Streamlit Secrets (or env vars).")
    st.stop()

# Build a session explicitly from the values above
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=(AWS_SESSION_TOKEN or None),
    region_name=AWS_REGION,
)
s3 = session.client("s3")

# Optional: STS sanity check to catch invalid/expired keys early
try:
    ident = session.client("sts").get_caller_identity()
    st.caption(f"STS OK: account {ident['Account']} • {ident['Arn']}")
except ClientError as e:
    st.error(f"STS check failed: {e}")
    st.stop()


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
