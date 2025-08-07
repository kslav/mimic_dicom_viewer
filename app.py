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
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing import Tuple

# ------ DICOM HELPERS ------
def first_value(x):
    # DICOM WindowCenter/Width can be MultiValue
    try:
        return float(x[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return None

def read_dicom_from_s3(bucket: str, key: str, s3_client) -> Tuple["pydicom.Dataset", np.ndarray]:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    ds = pydicom.dcmread(io.BytesIO(obj["Body"].read()))
    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale to get physical units (e.g., HU) if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return ds, arr

def window_to_float01(arr: np.ndarray, center: float, width: float, invert: bool) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    out = (arr - lo) / max(hi - lo, 1e-6)
    out = np.clip(out, 0.0, 1.0)
    if invert:
        out = 1.0 - out
    return out  # float in [0,1] — perfect for st.image

def dicom_to_uint8(ds):
    """Apply VOI LUT / windowing, handle MONOCHROME1, return 8-bit image."""
    try:
        data = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        data = ds.pixel_array

    data = data.astype(np.float32)

    # Invert if MONOCHROME1 (black/white reversed)
    if ds.get("PhotometricInterpretation", "").upper() == "MONOCHROME1":
        data = data.max() - data

    # Use WindowCenter/WindowWidth if available
    wc = ds.get("WindowCenter")
    ww = ds.get("WindowWidth")
    # Handle MultiValue
    if isinstance(wc, (list, tuple)) or getattr(wc, "__len__", None):
        wc = float(wc[0])
    if isinstance(ww, (list, tuple)) or getattr(ww, "__len__", None):
        ww = float(ww[0])

    if wc is not None and ww:
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
        data = np.clip(data, lo, hi)
        data = (data - lo) / max(hi - lo, 1e-6)
    else:
        # Fallback to min-max
        data = data - data.min()
        denom = data.max()
        data = data / denom if denom > 0 else data

    img8 = (data * 255.0).clip(0, 255).astype(np.uint8)
    return img8

# ---- GUI helpers ----
def first_value(x):
    if x is None:
        return None
    try:
        return float(x[0])  # DICOM MultiValue
    except Exception:
        try:
            return float(x)
        except Exception:
            return None

def pick_defaults(ds, arr):
    """Choose default WC/WW from DICOM tags if present, else robust percentiles; detect MONOCHROME1."""
    wc = first_value(getattr(ds, "WindowCenter", None))
    ww = first_value(getattr(ds, "WindowWidth", None))

    # Fallback if missing or invalid
    if wc is None or ww is None or ww <= 0:
        vmin, vmax = np.percentile(arr, [0.5, 99.5])
        wc = float((vmin + vmax) / 2.0)
        ww = float(max(vmax - vmin, 1.0))

    invert_default = (getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1")
    return wc, ww, invert_default

def slider_bounds(arr):
    """Give sane slider limits around the signal range."""
    vmin, vmax = np.percentile(arr, [0.5, 99.5])
    pad = 0.2 * max(vmax - vmin, 1.0)
    return float(vmin - pad), float(vmax + pad)

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
    EXPIRATION = int(os.getenv("PRESIGN_TTL", "3600"))  # 1 hour default
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
        The app will render a preview of each DICOM.
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

        for _, row in group.iterrows():
            uri = row["S3_PATH"]
            bucket, key = parse_s3_uri(uri)

            # Load DICOM + defaults
            ds, arr = read_dicom_from_s3(bucket, key, s3)
            wc_default, ww_default, invert_default = pick_defaults(ds, arr)

            # Sliders + controls (unique keys per object)
            vmin, vmax = slider_bounds(arr)
            c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.8, 0.6])
            wc = c1.slider(
                "Window center",
                min_value=vmin, max_value=vmax,
                value=float(wc_default), key=f"wc_{key}"
            )
            ww = c2.slider(
                "Window width",
                min_value=1.0, max_value=float(max(arr.max() - arr.min(), ww_default*2)),
                value=float(max(ww_default, 1.0)), key=f"ww_{key}"
            )
            invert = c3.checkbox("Invert", value=invert_default, key=f"invert_{key}")
            if c4.button("Reset", key=f"reset_{key}"):
                st.session_state[f"wc_{key}"] = float(wc_default)
                st.session_state[f"ww_{key}"] = float(max(ww_default, 1.0))
                st.session_state[f"invert_{key}"] = invert_default
                st.experimental_rerun()

            # Compute the view (float [0,1]) and display
            view = window_to_float01(arr, wc, ww, invert)
            st.image(
                view,
                caption=key.split("/")[-1],
                use_container_width=True,
                clamp=True,  # safe since view is in [0,1]
            )

    st.success("Done!")

if __name__ == "__main__":
    main()
