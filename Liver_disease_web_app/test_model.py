import torch

from utils.model_architectures import EfficientNet_HybridAttention


# ================================================================
# SETTINGS
# ================================================================

MODEL_PATH = "models/best_stage2_cutmix_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MODEL LOADING TEST")
print("=" * 70)

print(f"Device: {DEVICE}")


# ================================================================
# CREATE MODEL
# ================================================================

model = EfficientNet_HybridAttention(
    num_classes=4,
    num_heads=8
)

model = model.to(DEVICE)


# ================================================================
# LOAD CHECKPOINT
# ================================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Handle either a raw state_dict or a checkpoint dictionary
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


# ================================================================
# EVALUATION MODE
# ================================================================

model.eval()


# ================================================================
# PARAMETER COUNT
# ================================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Total parameters: {total_params:,}")


# ================================================================
# TEST FORWARD PASS
# ================================================================

dummy_input = torch.randn(
    1,
    3,
    224,
    224
).to(DEVICE)


with torch.no_grad():

    output = model(dummy_input)


print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")


# ================================================================
# FINAL CHECK
# ================================================================

if output.shape == (1, 4):

    print()
    print("SUCCESS!")
    print("Model loaded correctly.")
    print("Forward pass successful.")
    print("Output contains 4 classes.")

else:

    print()
    print("ERROR: Unexpected output shape.")


print("=" * 70)