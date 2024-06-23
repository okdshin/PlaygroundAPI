import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import Column, DateTime, Integer, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import AutoConfig

engine = create_engine("sqlite:///model_state.db", echo=True)
with engine.connect():
    pass

Base = declarative_base()


class ModelState(Base):
    __tablename__ = "model_state"

    id = Column(Integer, primary_key=True)
    model_name = Column(Text, nullable=False)
    last_accessed_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    deployment_id = Column(Text, unique=True, nullable=False)
    url = Column(Text, unique=True, nullable=False)

    deploy_method = Column(Text, nullable=False)
    n_gpu = Column(Integer, nullable=False)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

app = FastAPI()


def try_to_get_deployment_id(model_name: str, deploy_method: str) -> str | None:
    with Session() as session:
        model_state: ModelState | None = (
            session.query(ModelState)
            .filter_by(
                model_name=model_name,
                deploy_method=deploy_method,
            )
            .first()
        )
        print("try_to_get_deployment_id", model_state)
        if model_state is None:
            return None
        else:
            return model_state.deployment_id


def try_to_get_url(deployment_id: str) -> str | None:
    with Session() as session:
        model_state: ModelState | None = (
            session.query(ModelState).filter_by(deployment_id=deployment_id).first()
        )
        print("try_to_get_url", model_state)
        assert model_state is not None
        print("check", f"{model_state.url}/health")
        health_response = httpx.get(f"{model_state.url}/health")
        if health_response.status_code == 200:
            return model_state.url
        else:
            return None


def update_last_accessed_at(deployment_id: str) -> None:
    with Session() as session:
        model_state: ModelState | None = (
            session.query(ModelState).filter_by(deployment_id=deployment_id).first()
        )
        assert model_state is not None
        model_state.last_accessed_at = datetime.utcnow()
        session.commit()


@app.api_route("/{deploy_method}/v1/chat/completions", methods=["POST"])
async def chat_completions(deploy_method: str, request: Request):
    print("deploy_method", deploy_method)
    request_json = await request.json()
    if "model" not in request_json:
        raise HTTPException(status_code=400, detail='"model" attribute is not included')
    model_name = request_json["model"]
    print("model_name", model_name)

    deployment_id_or_none: str | None = try_to_get_deployment_id(
        model_name=model_name, deploy_method=deploy_method
    )
    print("deployment_id_or_none", deployment_id_or_none)
    if deployment_id_or_none is None:
        print("deploy", model_name)
        n_gpu: int = detect_suitable_computing_resource_from_model_name(
            model_name=model_name,
            deploy_method=deploy_method,
        )
        deployment_id: str = deploy_model(
            model_name=model_name, deploy_method=deploy_method, n_gpu=n_gpu
        )
    else:
        deployment_id: str = deployment_id_or_none
    print("deployment_id", deployment_id)

    print("check!")
    url_or_none: str | None = try_to_get_url(deployment_id=deployment_id)
    if url_or_none is None:
        print(model_name, "is not running yet")
        raise HTTPException(status_code=503, detail="model is not running yet")

    url: str = url_or_none
    print(model_name, "is running")
    update_last_accessed_at(deployment_id=deployment_id)
    print("post", f"{url}/v1/chat/completions")
    response = httpx.post(f"{url}/v1/chat/completions", json=request_json)
    print("response", response)
    return response.json()


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions_wo_deploy_method_specification(request: Request):
    return await chat_completions(deploy_method="vllm_fp16", request=request)


DELETE_UNUSED_MODEL_INTERVAL_SECONDS = int(
    os.getenv("PLAYGROUND_API_DELETE_UNUSED_MOODEL_INTERVAL_SECONDS", 60 * 60)
)


def delete_unused_model() -> None:
    now = datetime.utcnow()
    with Session() as session:
        model_state_list = session.query(ModelState).all()
        for model_state in model_state_list:
            if (
                now - model_state.last_accessed_at
            ).total_seconds() > DELETE_UNUSED_MODEL_INTERVAL_SECONDS:
                delete_deployment(deployment_id=model_state.deployment_id)
                session.delete(model_state)
        session.commit()


@app.on_event("startup")
def schedule_process() -> None:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        delete_unused_model,
        "interval",
        seconds=min(60, DELETE_UNUSED_MODEL_INTERVAL_SECONDS),
    )
    scheduler.start()


@app.get("/running_model_list")
def running_model_list():
    with Session() as session:
        model_state_list = session.query(ModelState).all()
    return model_state_list


def run(command: str, work_dir: str) -> None:
    print(f"+ {command}")
    subprocess.run(command, shell=True, cwd=work_dir, check=True)


def detect_model_size_in_giga_bytes(model_name: str) -> int:
    if Path(model_name).exists():
        params_file_list = list(Path(model_name).glob("**/*.safetensors")) or list(
            Path(model_name).glob("**/*.bin")
        )
        if len(params_file_list) == 0:
            raise HTTPException(status_code=400, detail="failed to detect model size")
        total_file_size_in_bytes = 0
        for params_file in params_file_list:
            file_size_in_bytes = os.path.getsize(params_file)
            total_file_size_in_bytes += file_size_in_bytes
    else:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                run(
                    f"GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{model_name}",
                    work_dir=tmp,
                )
            except Exception:
                raise HTTPException(
                    status_code=400, detail=f'failed to git clone model "{model_name}"'
                )
            lfs_alias_list = list(Path(tmp).glob("**/*.safetensors")) or list(
                Path(tmp).glob("**/*.bin")
            )
            if len(lfs_alias_list) == 0:
                raise HTTPException(
                    status_code=400, detail="failed to detect model size"
                )
            total_file_size_in_bytes = 0
            for lfs_alias in lfs_alias_list:
                with open(lfs_alias) as lfs_alias_file:
                    for line in lfs_alias_file:
                        if line.startswith("size"):
                            file_size_in_bytes = int(line.split()[1])
                            total_file_size_in_bytes += file_size_in_bytes
    return total_file_size_in_bytes / (1024**3)


def get_suitable_computing_resource(model_size_in_giga_bytes: float) -> int:
    usage_ratio = 0.9
    if model_size_in_giga_bytes < (24 * 6) * usage_ratio:
        n_gpu = int(model_size_in_giga_bytes / 24 + 1)
    else:
        raise HTTPException(status_code=400, detail="model is too big")
    if n_gpu > 1 and n_gpu % 2 == 1:
        n_gpu += 1
    assert n_gpu >= 1
    return n_gpu


def detect_suitable_computing_resource_from_model_name(
    model_name: str, deploy_method: str
) -> int:
    model_size_in_giga_bytes = detect_model_size_in_giga_bytes(model_name=model_name)
    n_gpu = get_suitable_computing_resource(
        model_size_in_giga_bytes=model_size_in_giga_bytes
    )
    return n_gpu


def is_model_name_valid(model_name: str) -> bool:
    if Path(model_name).exists():
        return True
    try:
        AutoConfig.from_pretrained(model_name)
        return True
    except Exception as e:
        print(e)
        return False


def deploy_model(model_name: str, deploy_method: str, n_gpu: int) -> str:
    if not is_model_name_valid(model_name=model_name):
        raise HTTPException(status_code=404, detail="{model_name} is not found")
    # TODO
    # f"python3 deploy.py --model-name {model_name} --n-gpu {n_gpu}"
    deployment_id = "hoge"
    url = "http://localhost:8080"
    with Session() as session:
        new_model_state = ModelState(
            model_name=model_name,
            deployment_id=deployment_id,
            url=url,
            deploy_method=deploy_method,
            n_gpu=n_gpu,
        )
        session.add(new_model_state)
        session.commit()
    return deployment_id


def delete_deployment(deployment_id: str) -> None:
    print("delete deployment", deployment_id)
    # TODO
