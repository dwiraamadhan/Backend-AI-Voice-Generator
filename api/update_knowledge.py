from fastapi import APIRouter, UploadFile, File, HTTPException
from functions.knowledge import add_knowledge


router = APIRouter()


@router.post("/knowledge")
async def update_knowledge(
    file: UploadFile = File(...),
):
    try:
        result = add_knowledge(file)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
