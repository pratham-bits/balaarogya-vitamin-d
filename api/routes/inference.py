from pathlib import Path
import json
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas.inference import (
    VitaminDInferenceRequest,
    VitaminDInferenceResponse,
)
from api.services.vitamin_d_service import (
    VitaminDInferenceService,
)
from src.models.model_loader import ModelArtifactLoader


router = APIRouter(
    prefix="/api/v1/vitamin-d",
    tags=["Vitamin-D"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "vitamin_d_api_baseline.joblib"
)


artifact = ModelArtifactLoader(
    ARTIFACT_PATH
).load()

service = VitaminDInferenceService(
    model=artifact["model"],
    metadata=artifact["metadata"],
    model_version=artifact["metadata"].model_version,
)


@router.post(
    "/assess",
    response_model=VitaminDInferenceResponse,
)
def assess_vitamin_d(
    request: VitaminDInferenceRequest,
) -> VitaminDInferenceResponse:
    return service.assess(request)


@router.post(
    "/assess-with-image",
    response_model=VitaminDInferenceResponse,
)
async def assess_vitamin_d_with_image(
    request: str = Form(...),
    image: UploadFile = File(...),
) -> VitaminDInferenceResponse:

    try:
        request_data = json.loads(request)
        inference_request = VitaminDInferenceRequest.model_validate(
            request_data
        )

    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid assessment request JSON.",
        ) from exc

    if not image.filename:
        raise HTTPException(
            status_code=422,
            detail="Image filename is missing.",
        )

    suffix = Path(image.filename).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=422,
            detail="Unsupported image format. Use JPG, JPEG, PNG, or WEBP.",
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temporary_path = Path(temp_file.name)

            while chunk := await image.read(1024 * 1024):
                temp_file.write(chunk)

        inference_request.image_path = str(temporary_path)

        return service.assess(inference_request)

    finally:
        await image.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)