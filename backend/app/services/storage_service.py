from __future__ import annotations

import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from app.config import Settings, get_settings


@dataclass(frozen=True)
class StoredObject:
    storage_uri: str
    relative_path: str
    size_bytes: int
    sha256: str
    mime_type: str | None


class StorageService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.backend = self.settings.storage_backend.lower()
        self.bucket = self.settings.s3_bucket
        self.local_root = Path(self.settings.storage_local_root)
        self._s3_client = None

    @property
    def s3_client(self):
        if self._s3_client is None:
            import boto3
            from botocore.client import Config

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint,
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                config=Config(signature_version="s3v4"),
                use_ssl=self.settings.s3_secure,
            )
        return self._s3_client

    def ensure_bucket(self) -> None:
        if self.backend != "s3":
            self.local_root.mkdir(parents=True, exist_ok=True)
            return
        buckets = self.s3_client.list_buckets().get("Buckets", [])
        if not any(bucket.get("Name") == self.bucket for bucket in buckets):
            self.s3_client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, relative_path: str, data: bytes, mime_type: str | None = None) -> StoredObject:
        clean_path = relative_path.strip("/").replace("\\", "/")
        digest = hashlib.sha256(data).hexdigest()
        mime = mime_type or mimetypes.guess_type(clean_path)[0] or "application/octet-stream"

        if self.backend == "s3":
            self.ensure_bucket()
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=clean_path,
                Body=data,
                ContentType=mime,
                Metadata={"sha256": digest},
            )
            storage_uri = f"s3://{self.bucket}/{clean_path}"
        else:
            target = self.local_root / clean_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            storage_uri = f"local://{clean_path}"

        return StoredObject(
            storage_uri=storage_uri,
            relative_path=clean_path,
            size_bytes=len(data),
            sha256=digest,
            mime_type=mime,
        )

    def put_file(self, relative_path: str, source_path: str | Path, mime_type: str | None = None) -> StoredObject:
        source = Path(source_path)
        clean_path = relative_path.strip("/").replace("\\", "/")
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        mime = mime_type or mimetypes.guess_type(source.name)[0] or "application/octet-stream"

        if self.backend == "s3":
            self.ensure_bucket()
            self.s3_client.upload_file(
                str(source),
                self.bucket,
                clean_path,
                ExtraArgs={"ContentType": mime, "Metadata": {"sha256": digest.hexdigest()}},
            )
            storage_uri = f"s3://{self.bucket}/{clean_path}"
        else:
            target = self.local_root / clean_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            storage_uri = f"local://{clean_path}"

        return StoredObject(
            storage_uri=storage_uri,
            relative_path=clean_path,
            size_bytes=size,
            sha256=digest.hexdigest(),
            mime_type=mime,
        )

    def get_bytes(self, relative_path: str) -> bytes:
        clean_path = relative_path.strip("/").replace("\\", "/")
        if self.backend == "s3":
            response = self.s3_client.get_object(Bucket=self.bucket, Key=clean_path)
            return response["Body"].read()
        return (self.local_root / clean_path).read_bytes()

    def download_to_file(self, relative_path: str, target_path: str | Path) -> Path:
        clean_path = relative_path.strip("/").replace("\\", "/")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "s3":
            self.s3_client.download_file(self.bucket, clean_path, str(target))
        else:
            shutil.copyfile(self.local_root / clean_path, target)
        return target

    def delete(self, relative_path: str) -> None:
        clean_path = relative_path.strip("/").replace("\\", "/")
        if self.backend == "s3":
            self.s3_client.delete_object(Bucket=self.bucket, Key=clean_path)
            return
        target = self.local_root / clean_path
        if target.exists():
            target.unlink()

    def delete_uri(self, storage_uri: str) -> None:
        if storage_uri.startswith("s3://"):
            bucket_and_key = storage_uri.removeprefix("s3://")
            bucket, _, key = bucket_and_key.partition("/")
            if key:
                self.s3_client.delete_object(Bucket=bucket or self.bucket, Key=key)
            return
        if storage_uri.startswith("local://"):
            self.delete(storage_uri.removeprefix("local://"))

    def open_download(self, relative_path: str):
        clean_path = relative_path.strip("/").replace("\\", "/")
        if self.backend == "s3":
            response = self.s3_client.get_object(Bucket=self.bucket, Key=clean_path)
            return response["Body"]
        return (self.local_root / clean_path).open("rb")

    def presigned_url(
        self,
        relative_path: str,
        expires_in: int = 3600,
        response_content_disposition: str | None = None,
    ) -> str | None:
        clean_path = relative_path.strip("/").replace("\\", "/")
        if self.backend != "s3":
            return None
        params = {"Bucket": self.bucket, "Key": clean_path}
        if response_content_disposition:
            params["ResponseContentDisposition"] = response_content_disposition
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    def public_api_url(self, artifact_id: str, preview: bool = False) -> str:
        suffix = "preview" if preview else "download"
        return f"{self.settings.api_v1_prefix}/artifacts/{quote(artifact_id)}/{suffix}"
