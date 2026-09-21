import io
import json

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi import HTTPException
import pytest

from nebulax.api import r2_store, run_store


@pytest.fixture
def storage():
    s3 = boto3.client('s3', region_name='auto', endpoint_url='https://example.r2.cloudflarestorage.com',
                      aws_access_key_id='test', aws_secret_access_key='test')
    with Stubber(s3) as stub:
        yield r2_store.Bucket('test-bucket', s3), stub
        stub.assert_no_pending_responses()


def body(data):
    return StreamingBody(io.BytesIO(data), len(data))


def test_create_only_and_etag_pinned_reads(storage):
    bucket, stub = storage
    stub.add_response('put_object', {}, {'Bucket': bucket.name, 'Key': 'input', 'Body': b'raw',
                                        'ContentType': 'application/octet-stream', 'IfNoneMatch': '*'})
    blob = bucket.blob('input')
    blob.upload_from_string(b'raw', if_generation_match=0)
    stub.add_response('get_object', {'Body': body(b'raw')},
                      {'Bucket': bucket.name, 'Key': 'input', 'IfMatch': 'etag-1'})
    target = io.BytesIO()
    blob.download_to_file(target, if_generation_match='etag-1')
    assert target.getvalue() == b'raw'
    stub.add_client_error('get_object', 'PreconditionFailed', http_status_code=412,
                          expected_params={'Bucket': bucket.name, 'Key': 'input', 'IfMatch': 'old'})
    with pytest.raises(r2_store.StorageError) as exc:
        blob.download_to_file(io.BytesIO(), if_generation_match='old')
    assert exc.value.code == 412


def test_storage_missing_and_forbidden_differ(storage):
    bucket, stub = storage
    for code in (404, 403):
        stub.add_client_error('head_object', str(code), http_status_code=code,
                              expected_params={'Bucket': bucket.name, 'Key': 'input'})
    assert not bucket.blob('input').exists()
    with pytest.raises(r2_store.StorageError) as exc:
        bucket.blob('input').exists()
    assert exc.value.code == 403


def resume_setup(bucket, stub, size):
    blob = bucket.blob('input')
    stub.add_client_error('head_object', '404', http_status_code=404,
                          expected_params={'Bucket': bucket.name, 'Key': 'input'})
    state = r2_store.state_blob(blob)
    stub.add_response('head_object', {'ContentLength': 20, 'ETag': 'state'},
                      {'Bucket': bucket.name, 'Key': state.name})
    stub.add_response('get_object', {'Body': body(json.dumps({'upload_id': 'upload-1'}).encode())},
                      {'Bucket': bucket.name, 'Key': state.name})
    return blob


def test_resume_uses_persisted_session_and_accepted_parts(storage):
    bucket, stub = storage
    blob = resume_setup(bucket, stub, r2_store.CHUNK + 10)
    stub.add_response('list_parts', {'Parts': [{'PartNumber': 1, 'Size': r2_store.CHUNK, 'ETag': 'p1'}]},
                      {'Bucket': bucket.name, 'Key': 'input', 'UploadId': 'upload-1'})
    plan = r2_store.session(blob, r2_store.CHUNK + 10)
    assert plan['provider'] == 'r2'
    assert plan['parts'][0]['done'] is True
    assert plan['parts'][1]['done'] is False
    assert plan['parts'][1]['size'] == 10
    assert 'uploadId=upload-1' in plan['parts'][1]['url']


def test_complete_rejects_wrong_part_sizes(storage):
    bucket, stub = storage
    blob = resume_setup(bucket, stub, 10)
    stub.add_response('list_parts', {'Parts': [{'PartNumber': 1, 'Size': 11, 'ETag': 'p1'}]},
                      {'Bucket': bucket.name, 'Key': 'input', 'UploadId': 'upload-1'})
    with pytest.raises(HTTPException) as exc:
        r2_store.complete(blob, 10)
    assert exc.value.status_code == 409


def test_complete_uses_server_list_and_is_idempotent(storage):
    bucket, stub = storage
    blob = resume_setup(bucket, stub, 10)
    stub.add_response('list_parts', {'Parts': [{'PartNumber': 1, 'Size': 10, 'ETag': 'p1'}]},
                      {'Bucket': bucket.name, 'Key': 'input', 'UploadId': 'upload-1'})
    stub.add_response('complete_multipart_upload', {}, {'Bucket': bucket.name, 'Key': 'input',
                      'UploadId': 'upload-1', 'MultipartUpload': {'Parts': [{'PartNumber': 1, 'ETag': 'p1'}]}})
    for _ in range(2):
        stub.add_response('head_object', {'ContentLength': 10, 'ETag': 'done'},
                          {'Bucket': bucket.name, 'Key': 'input'})
    assert r2_store.complete(blob, 10)['complete']
    assert r2_store.complete(blob, 10)['complete']


def test_provider_selection(monkeypatch):
    monkeypatch.setenv('NEBULAX_RUN_BUCKET', 'archive')
    monkeypatch.setenv('NEBULAX_STORAGE_PROVIDER', 'r2')
    sentinel = object()
    monkeypatch.setattr(r2_store, 'client', lambda: sentinel)
    assert run_store.bucket().s3 is sentinel
    monkeypatch.setenv('NEBULAX_STORAGE_PROVIDER', 'typo')
    with pytest.raises(HTTPException):
        run_store.bucket()
