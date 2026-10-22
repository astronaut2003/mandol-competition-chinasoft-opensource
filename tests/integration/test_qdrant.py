"""Opt-in real Qdrant transport/tenant filtering with synthetic vectors.

This verifies storage protocol only; vectors are explicitly synthetic and do
not certify Qwen, SPLADE, or BGE inference.
"""
import os,uuid
from urllib.parse import urlsplit
import pytest
from qdrant_client import QdrantClient,models

@pytest.fixture
def server():
 url=os.getenv('MANDOL_TEST_QDRANT')
 if not url:pytest.skip('Set MANDOL_TEST_QDRANT to your independent test server')
 parsed=urlsplit(url)
 if parsed.hostname in ('localhost','127.0.0.1') and parsed.port in (18633,19633):pytest.fail('Reserved local deployment cannot be used as a release test server')
 client=QdrantClient(url=url,timeout=10,prefer_grpc=False,check_compatibility=False)
 name='chinasoft_release_test_'+uuid.uuid4().hex
 client.create_collection(name,vectors_config={'dense':models.VectorParams(size=4,distance=models.Distance.COSINE)},sparse_vectors_config={'sparse':models.SparseVectorParams()})
 ids=[str(uuid.uuid4()) for _ in range(3)]
 client.upsert(name,points=[models.PointStruct(id=uid,vector={'dense':[1.,0.,0.,0.],'sparse':models.SparseVector(indices=[12,300],values=[1.,.5])},payload={'tenant':tenant,'active':True}) for uid,tenant in zip(ids,['a','b','a'])],wait=True)
 try:yield client,name,ids
 finally:
  client.delete_collection(name);client.close()

def tenant(value):return models.Filter(must=[models.FieldCondition(key='tenant',match=models.MatchValue(value=value))])

@pytest.mark.integration_qdrant
def test_real_dense_sparse_schema_and_tenant_filter(server):
 c,name,ids=server
 dense=c.query_points(name,query=[1.,0.,0.,0.],using='dense',query_filter=tenant('a'),with_payload=True).points
 sparse=c.query_points(name,query=models.SparseVector(indices=[12],values=[1.]),using='sparse',query_filter=tenant('b'),with_payload=True).points
 assert {str(p.id) for p in dense}=={ids[0],ids[2]}
 assert [str(p.id) for p in sparse]==[ids[1]] and sparse[0].payload['tenant']=='b'
 assert c.get_collection(name).points_count==3

@pytest.mark.integration_qdrant
def test_real_delete_and_reopen_preserve_scoped_records(server):
 c,name,ids=server
 c.delete(name,points_selector=models.PointIdsList(points=[ids[0]]),wait=True)
 reopened=QdrantClient(url=os.environ['MANDOL_TEST_QDRANT'],check_compatibility=False)
 try:
  rows=reopened.query_points(name,query=[1.,0.,0.,0.],using='dense',query_filter=tenant('a')).points
  assert [str(p.id) for p in rows]==[ids[2]]
  assert reopened.retrieve(name,[ids[1]])[0].payload['tenant']=='b'
 finally:reopened.close()
