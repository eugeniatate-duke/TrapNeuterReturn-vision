import pandas as pd
import streamlit as st

from PIL import Image

from app.predictor import Predictor
from app.recommendations import (
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


def run_app():

    predictor = Predictor()

    st.title("🐾 TNRVision")

    st.subheader(
        "AI-Assisted Trap Monitoring"
    )

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

    uploaded_file = st.file_uploader(
        "Upload trap camera image",
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded_file:
        return

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

    recommendation = (
        get_recommendation(
            result["animal"],
            result["confidence"],
        )
    )

    st.subheader(
        "Recommended Action"
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

    if result["confidence"] >= 0.90:

        confidence_level = "High"

    elif result["confidence"] >= 0.70:

        confidence_level = "Moderate"

    else:

        confidence_level = "Low"

    st.info(
        f"Confidence Level: {confidence_level}"
    )

    if result["confidence"] < 0.70:

        st.error(
            "🔍 Human Review Required"
        )

        st.write(
            recommendation
        )

    else:

        st.success(
            recommendation
        )

    probs = pd.DataFrame(
        {
            "Animal":
                result["probabilities"].keys(),

            "Probability":
                result["probabilities"].values(),
        }
    )

    st.subheader(
        "Prediction Probabilities"
    )

    st.bar_chart(
        probs.set_index(
            "Animal"
        )
    )