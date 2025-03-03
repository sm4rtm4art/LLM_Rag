from fastapi import APIRouter, status

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for Kubernetes probes.

    Returns 200 OK when the service is healthy.
    """
    return {"status": "healthy"}
