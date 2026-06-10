RECOMMENDATIONS = {
    "cat":
        "Proceed with TNR evaluation.",

    "dog":
        "Do not trap. Likely domestic animal.",

    "raccoon":
        "Do not trap. Non-target wildlife detected.",

    "opossum":
        "Release if trapped. Non-target wildlife detected."
}


def get_recommendation(
    animal,
    confidence,
):
    if confidence < 0.70:
        return (
            "Manual review required "
            "before taking action."
        )

    return RECOMMENDATIONS.get(
        animal,
        "Manual review required."
    )