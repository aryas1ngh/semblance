"""Streamlit UI: upload a product image, see the most similar gallery items."""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Semblance", layout="wide")
st.title("Semblance — visual product search")
st.caption("Upload a product photo; the model retrieves the most visually similar items from the gallery.")

k = st.sidebar.slider("Results", 4, 24, 12, step=4)
upload = st.file_uploader("Product image", type=["jpg", "jpeg", "png"])

if upload:
    st.image(upload, caption="query", width=250)
    try:
        resp = requests.post(
            f"{API}/search",
            files={"file": (upload.name, upload.getvalue(), upload.type)},
            params={"k": k},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"search failed: {e}")
        st.stop()

    results = resp.json()["results"]
    st.subheader(f"Top {len(results)} matches")
    cols = st.columns(4)
    for n, r in enumerate(results):
        # fetch the thumbnail through this server so browser only talks to Streamlit
        thumb = requests.get(f"{API}/image", params={"path": r["path"]}, timeout=10).content
        with cols[n % 4]:
            st.image(thumb, use_container_width=True)
            st.caption(f"#{r['rank']} · {r['category']} · score {r['score']:.3f}")
