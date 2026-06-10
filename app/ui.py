import pandas as pd

import streamlit as st

from PIL import Image

from predictor import Predictor

from recommendations import (
    get_recommendation,
)

TARGET_SPECIES = [
    "cat"
]

NON_TARGET_SPECIES = [
    "dog",
    "raccoon",
    "opossum"
]

with st.expander(
    "About TNRVision"
):

    st.write(
        """
        TNRVision is an AI-assisted trap
        monitoring system designed to
        support Trap-Neuter-Return
        programs.

        The system identifies common
        target and non-target species
        encountered during trap
        monitoring and provides
        operational recommendations.
        """
    )

st.set_page_config(
    page_title="TNRVision",
    layout="wide",
)

predictor = Predictor()

st.title(
    "🐾 TNRVision"
)

st.subheader(
    "AI-Assisted Trap Monitoring"
)

uploaded_file = st.file_uploader(
    "Upload trap camera image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        width=400,
    )

    result = predictor.predict(
        image
    )

    st.subheader("Recommended Action")
    
    recommendation = (
        get_recommendation(
            result["animal"],
            result["confidence"],
        )
    )

    if result["animal"] in TARGET_SPECIES:
        st.success(
            "🎯 Target Species Detected"
        )

    else:
        st.warning(
            "⚠️ Non-Target Species Detected"
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Animal Detected",
            result["animal"].title(),
        )

    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
        )

    # st.success(
    #     recommendation
    # )

    if result["confidence"] < 0.70:
        st.error("🔍 Human Review Required")
        st.write(recommendation)

    else:
        st.success(recommendation)

    probs = pd.DataFrame(
        {
            "Animal":
                result[
                    "probabilities"
                ].keys(),
            "Probability":
                result[
                    "probabilities"
                ].values(),
        }
    )

    st.subheader("Prediction Probabilities")

    st.bar_chart(probs.set_index("Animal"))

    if result["confidence"] >= 0.90:
        confidence_level = "High"

    elif result["confidence"] >= 0.70:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    st.info(f"Confidence Level: {confidence_level}")