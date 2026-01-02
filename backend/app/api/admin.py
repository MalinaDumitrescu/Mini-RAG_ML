from fastapi import APIRouter

router = APIRouter()

@router.post("/rebuild_index")
def rebuild_index():
    return {"status": "Index rebuild started"}
