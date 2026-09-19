import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from utils.model_architectures import EfficientNet_HybridAttention
from utils.gradcam import GradCAM

# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Liver Image Classification",
    page_icon="🔬",
    layout="wide"
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 10px;
    }

    .disclaimer {
        font-size: 13px;
        padding: 12px;
        border-radius: 8px;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ================================================================
# CONSTANTS
# ================================================================

CLASS_NAMES = [
    "Ballooning",
    "Fibrosis",
    "Inflammation",
    "Steatosis"
]

MODEL_PATH = "models/best_stage2_cutmix_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# IMAGE TRANSFORMATION
# ================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource
def load_model():

    model = EfficientNet_HybridAttention(
        num_classes=4,
        num_heads=8
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ================================================================
# LOAD MODEL
# ================================================================

model = load_model()


# ================================================================
# TITLE
# ================================================================

st.markdown(
    '<div class="main-title">🔬 Liver Image Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based liver histopathology image classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:

    st.header("🔬 Model Information")

    st.write("### Current Model")

    st.write(
        "**Model 3 — Proposed Architecture**"
    )

    st.write(
        "EfficientNet-B3 → Channel Attention "
        "→ Spatial Attention → MHSA"
    )

    st.divider()

    st.write("### Configuration")

    st.write("**Input size:** 224 × 224")

    st.write("**Classes:** 4")

    st.write(
        "**Device:** "
        f"{DEVICE}"
    )

    st.divider()

    st.write("### Classes")

    for class_name in CLASS_NAMES:

        st.write(
            f"• {class_name}"
        )

    st.divider()

    st.info(
        "Research and educational use only. "
        "This model is not a clinical diagnostic system."
    )

# ================================================================
# IMAGE UPLOAD
# ================================================================

uploaded_file = st.file_uploader(
    "Upload a liver image",
    type=["jpg", "jpeg", "png"]
)


# ================================================================
# PREDICTION
# ================================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # ================================================================
    # IMAGE + PREDICTION LAYOUT
    # ================================================================

    image_col, result_col = st.columns(
        [1, 1]
    )

    with image_col:

        st.subheader("Uploaded Image")

        st.image(
            image,
            width=400
        )


    with result_col:

        st.subheader("Prediction")

        # The actual prediction is calculated below.
    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # ------------------------------------------------------------
    # MODEL INFERENCE
    # ------------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

    probabilities = probabilities[0].cpu()

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ].item() * 100


    # ============================================================
    # RESULTS
    # ============================================================

    st.divider()

    st.subheader("Prediction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Predicted Class",
            predicted_class
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with result_col2:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ============================================================
    # CLASS PROBABILITIES
    # ============================================================

    st.subheader(
        "Class Probabilities"
    )

    probability_dict = {
        CLASS_NAMES[i]: float(
            probabilities[i]
        )
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(
        probability_dict
    )


# ================================================================
# GRAD-CAM EXPLAINABILITY
# ================================================================

st.divider()

st.subheader("Explainable AI — Grad-CAM")

st.write(
    "The heatmap highlights image regions that contributed "
    "to the selected model prediction."
)


if st.button("Generate Grad-CAM"):

    model.zero_grad()

    target_layer = model.features[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    heatmap = gradcam.generate(
        input_tensor,
        target_class=predicted_index
    )

    import numpy as np
    import matplotlib.pyplot as plt

    heatmap_np = heatmap.numpy()

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(image)

    ax.imshow(
        heatmap_np,
        alpha=0.45,
        cmap="jet"
    )

    ax.axis("off")

    # Center the visualization
    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.pyplot(
            fig,
            use_container_width=True
        )

    plt.close(fig)

    gradcam.remove_hooks()


st.divider()

st.caption(
    "Liver Image Classification | "
    "Research & Educational Prototype"
)

st.caption(
    "Model predictions should not be interpreted "
    "as medical diagnoses."
)