import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import (GPU_KINDS, RESOLUTIONS, RTX_MODELS, build_features,  # noqa: E402
                     default_cpu_brand, fair_price_check, load_artifacts)

st.set_page_config(page_title="Laptop Price Checker",
                   page_icon="💻", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


model, meta = get_artifacts()
cats = meta["categorical"]


def pick(options, preferred):
    return options.index(preferred) if preferred in options else 0


st.title("💻 Laptop Price Checker")
st.caption("Enter a laptop's specifications and the asking price. The model estimates a market price "
           "(Smartprix listings, ₹INR) and tells you whether the asking price looks fair.")

tab_check, tab_about = st.tabs(["Price check", "About the model"])

with tab_check:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Specifications")
        brand = st.selectbox("Brand", cats["brand_name"], index=pick(
            cats["brand_name"], "Lenovo"))
        cpu_series = st.selectbox(
            "Processor series", cats["iter"], index=pick(cats["iter"], "Core i5"))
        cpu_brand_default = default_cpu_brand(cpu_series)
        cpu_brand = st.selectbox("Processor brand", cats["processor_brand"],
                                 index=pick(cats["processor_brand"], cpu_brand_default))

        c1, c2 = st.columns(2)
        ram = c1.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64], index=3)
        storage_labels = {128: "128 GB", 256: "256 GB",
                          512: "512 GB", 1000: "1 TB", 2000: "2 TB"}
        storage = c2.selectbox("Storage", list(storage_labels), index=2,
                               format_func=lambda v: storage_labels[v])

        c3, c4 = st.columns(2)
        screen = c3.slider("Screen size (inches)", 11.6, 18.0, 15.6, 0.1)
        resolution = c4.selectbox("Resolution", list(RESOLUTIONS), index=1)
        touch = st.checkbox("Touchscreen")

        gpu_kind = st.radio("Graphics", GPU_KINDS, horizontal=True)
        rtx_model, vram, gpu_brand = 0, 0, "Intel"
        if gpu_kind == "Integrated":
            integrated_brands = [b for b in cats["gpu_brand"] if b != "NVIDIA"]
            gpu_brand = st.selectbox("Integrated GPU brand", integrated_brands,
                                     index=pick(integrated_brands, cpu_brand_default))
        elif gpu_kind == "NVIDIA RTX (dedicated)":
            g1, g2 = st.columns(2)
            rtx_model = g1.selectbox("RTX model", RTX_MODELS, index=RTX_MODELS.index(4060),
                                     format_func=lambda v: f"RTX {v}")
            vram = g2.selectbox("VRAM (GB)", [4, 6, 8, 12, 16, 24], index=2)
        else:
            g1, g2 = st.columns(2)
            gpu_brand = g1.selectbox("GPU brand", ["NVIDIA", "AMD"])
            vram = g2.selectbox("VRAM (GB)", [2, 4, 6, 8, 12, 16], index=1)

        with st.expander("More options (defaults are typical values)"):
            m = meta["numeric"]
            storage_type = st.selectbox("Storage type", cats["memory_type"],
                                        index=pick(cats["memory_type"], "SSD"))
            os_name = st.selectbox(
                "Operating system", cats["os"], index=pick(cats["os"], "Windows 11"))
            cores = st.number_input("CPU cores", int(m["core_num"]["min"]), int(m["core_num"]["max"]),
                                    int(m["core_num"]["median"]))
            threads = st.number_input("CPU threads", int(m["threads_num"]["min"]),
                                      int(m["threads_num"]["max"]), int(m["threads_num"]["median"]))
            warranty = st.selectbox("Warranty (years)", [1, 2, 3], index=0)

    features = build_features(
        brand=brand, cpu_brand=cpu_brand, cpu_series=cpu_series, cores=cores, threads=threads,
        ram_gb=ram, storage_gb=storage, storage_type=storage_type, screen_in=screen,
        resolution=resolution, touch=touch, gpu_kind=gpu_kind, rtx_model=rtx_model,
        vram_gb=vram, gpu_brand=gpu_brand, os_name=os_name, warranty=warranty)

    with right:
        st.subheader("Result")
        asking = st.number_input("Asking price (₹)", min_value=5000, max_value=1000000,
                                 value=70000, step=1000)
        res = fair_price_check(model, meta, features, asking)
        if gpu_kind != "Integrated" and cpu_brand in ("Qualcomm", "Apple"):
            st.info("Heads-up: laptops with this processor and a dedicated GPU don't appear in "
                    "the training data, so this estimate is unreliable.")

        k1, k2, k3 = st.columns(3)
        k1.metric("Model price", f"₹{res['predicted']:,}")
        k2.metric(
            "Fair range", f"₹{res['fair_low'] / 1000:,.0f}k–{res['fair_high'] / 1000:,.0f}k")
        k3.metric("Asking price", f"₹{asking:,}", delta=f"{res['diff_pct']:+.0f}% vs model",
                  delta_color="inverse")
        st.caption(
            f"Exact fair range: ₹{res['fair_low']:,} – ₹{res['fair_high']:,}")

        if res["verdict"] == "fair":
            st.success(
                "Fair: the asking price is inside the typical range for this kind of laptop.")
        elif res["verdict"] == "overpriced":
            st.error("Looks overpriced: above the typical range for these specs.")
        else:
            st.warning("Looks underpriced: below the typical range. It could be a good deal, "
                       "but check what the listing leaves out.")

        fig, ax = plt.subplots(figsize=(7, 1.8))
        ax.barh([0], [res["fair_high"] - res["fair_low"]], left=res["fair_low"], height=0.4,
                color="#cfe3d4", label="Fair range")
        ax.axvline(res["predicted"], color="green", lw=2, label="Model price")
        ax.axvline(asking, color="crimson", lw=2,
                   ls="--", label="Asking price")
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(lambda x, _: f"₹{x / 1000:,.0f}k")
        ax.legend(loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, 1.45), frameon=False)
        fig.tight_layout()
        st.pyplot(fig)

        st.caption("The fair range is the 10th to 90th percentile of actual/predicted price ratios "
                   "for listings in the same predicted-price tier (out-of-fold). It is wider for "
                   "expensive laptops, where the model is less precise.")

with tab_about:
    st.subheader("How it works")
    st.markdown(
        "- **Data:** 940 laptop listings from Smartprix (₹), after removing 78 repeated listings.\n"
        "- **Model:** XGBoost predicting log price, compared against Ridge and Random Forest.\n"
        "- **Validation:** 5-fold GroupKFold by laptop model name, so the same laptop never "
        "appears in both train and test.\n"
        "- **Explainability:** SHAP values show which specs push the price up or down.")

    if "cv_results" in meta:
        st.subheader("Cross-validated results")
        st.dataframe(pd.DataFrame(meta["cv_results"]).T)

    for name, caption in [("predicted_vs_actual.png", "Predicted vs actual price (XGBoost)"),
                          ("shap_importance.png", "Feature importance (SHAP)")]:
        path = ROOT / "reports" / "figures" / name
        if path.exists():
            st.image(str(path), caption=caption)

    st.subheader("Limitations")
    st.markdown(
        "- Prices are Smartprix listing prices from a single snapshot, not final sale prices.\n"
        "- The model does not know about condition, age or seller, so it does not apply to used "
        "or refurbished laptops.\n"
        "- It is least precise for the most expensive laptops.\n"
        "- Treat the verdict as a guide, not a guarantee.")
