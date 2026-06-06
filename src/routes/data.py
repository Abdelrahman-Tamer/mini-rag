from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request, Form
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController, NLPController
import aiofiles
from models import ResponseSignal
import logging
from .schemes.data import ProcessRequest, UpdateAssetRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemes import DataChunk, Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from typing import Optional

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str, file: UploadFile,
                    city: Optional[str] = Form(None),
                    doc_type: Optional[str] = Form(None),
                    app_settings: Settings = Depends(get_settings)):
        
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    # detect city and doc_type from filename if not provided
    if not city or not doc_type:
        extracted = data_controller.extract_city_type(filename=file.filename)
        if not city:
            city = extracted.get("city") or app_settings.DEFAULT_CITY
        if not doc_type:
            doc_type = extracted.get("doc_type") or app_settings.DEFAULT_DOC_TYPE

    # store the assets into the database
    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
        asset_config={"city": city, "doc_type": doc_type}
    )

    asset_record = await asset_model.create_asset(asset=asset_resource)

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": str(asset_record.id),
                "city": city,
                "doc_type": doc_type,
            }
        )

@data_router.put("/asset/{asset_id}")
async def update_asset(request: Request, asset_id: str, update_request: UpdateAssetRequest):

    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    asset = await asset_model.update_asset_city(
        asset_id=asset_id,
        city=update_request.city,
        doc_type=update_request.doc_type
    )

    if not asset:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_ID_ERROR.value}
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.CITY_UPDATE_SUCCESS.value,
            "asset_id": str(asset.id),
            "city": asset.asset_config.get("city") if asset.asset_config else None,
            "doc_type": asset.asset_config.get("doc_type") if asset.asset_config else None,
        }
    )


@data_router.get("/assets/uncategorized")
async def get_uncategorized_assets(request: Request):

    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    assets = await asset_model.get_uncategorized_assets()

    return JSONResponse(
        content={
            "assets": [
                {
                    "id": str(a.id),
                    "name": a.asset_name,
                    "project_id": str(a.asset_project_id),
                }
                for a in assets
            ]
        }
    )


@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    asset_model = await AssetModel.create_instance(
            db_client=request.app.db_client
        )

    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=process_request.file_id
        )

        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                }
            )

        project_files_ids = {
            asset_record.id: asset_record.asset_name
        }
    
    else:
        

        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
        )

        project_files_ids = {
            record.id: record.asset_name
            for record in project_files
        }

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            }
        )
    
    process_controller = ProcessController(project_id=project_id)

    no_records = 0
    no_files = 0

    chunk_model = await ChunkModel.create_instance(
                        db_client=request.app.db_client
                    )

    if do_reset == 1:
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.id
        )

    for asset_id, file_id in project_files_ids.items():

        # fetch asset record for metadata (city, doc_type)
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=file_id
        )
        chunk_metadata = {}
        if asset_record and asset_record.asset_config:
            chunk_metadata = {
                "city": asset_record.asset_config.get("city"),
                "doc_type": asset_record.asset_config.get("doc_type"),
                "asset_id": str(asset_record.id),
            }

        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.error(f"Error while processing file: {file_id}")
            continue

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            metadata=chunk_metadata
        )

        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.PROCESSING_FAILED.value
                }
            )

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i+1,
                chunk_project_id=project.id,
                chunk_asset_id=asset_id
            )
            for i, chunk in enumerate(file_chunks)
        ]

        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files += 1

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files
        }
    )


@data_router.delete("/project/{project_id}")
async def delete_project(request: Request, project_id: str):

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
        )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    _ = await chunk_model.delete_chunks_by_project_id(project_id=project.id)

    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    _ = await asset_model.delete_assets_by_project_id(project_id=project.id)

    _ = nlp_controller.delete_project_vectors(project_id=project_id)

    _ = await project_model.delete_project(project_id=project_id)

    return JSONResponse(
        content={"signal": ResponseSignal.DELETE_SUCCESS.value}
    )


@data_router.delete("/file/{file_id}")
async def delete_file(request: Request, file_id: str):

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    asset = await asset_model.delete_asset_by_id(asset_id=file_id)

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    _ = await chunk_model.delete_chunks_by_asset_id(asset_id=file_id)

    _ = nlp_controller.delete_file_vectors(asset_id=file_id)

    return JSONResponse(
        content={"signal": ResponseSignal.DELETE_SUCCESS.value}
    )


@data_router.get("/cities")
async def get_cities(request: Request):

    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    cities = await asset_model.collection.distinct(
        "asset_config.city",
        {"asset_config.city": {"$nin": [None, "unknown"]}}
    )

    return JSONResponse(
        content={"cities": [c for c in cities if c]}
    )


from pydantic import BaseModel

class CityRequest(BaseModel):
    city_name: str

@data_router.post("/cities")
async def add_city(request: Request, body: CityRequest):

    return JSONResponse(
        content={"signal": "city_added", "city": body.city_name}
    )


@data_router.delete("/chunks/{project_id}")
async def delete_chunks(request: Request, project_id: str):

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
        )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    _ = await chunk_model.delete_chunks_by_project_id(project_id=project.id)

    _ = nlp_controller.delete_project_vectors(project_id=project_id)

    return JSONResponse(
        content={"signal": ResponseSignal.DELETE_SUCCESS.value}
    )
