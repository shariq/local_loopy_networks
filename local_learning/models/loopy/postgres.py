import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, Float, PickleType, DateTime
from sqlalchemy.dialects import postgresql
import os
import datetime

connection_metas = {}

def initialize(postgres_uri):
    connection = sqlalchemy(postgres_uri, client_encoding='utf8')
    meta = sqlalchemy.MetaData(bind=connection, reflect=True)

    if 'results' not in meta.tables:
        results = Table('results', meta,
            Column('timestamp', DateTime),
            Column('score', Float),
            Column('code', String),
            Column('blob', PickleType)
        )

        # Create the above tables
        meta.create_all(con)

    connection_metas[postgres_uri] = (connection, meta)

    return connection, meta

def save(postgres_uri, score, code, blob):
    if postgres_uri not in connection_metas or connection_metas[postgres_uri][0].closed:
        connection, meta = initialize(postgres_uri)
    connection, meta = connection_metas[postgres_uri]
    timestamp = datetime.datetime.now()
    clause = meta.tables['results'].insert().values(
        timestamp=timestamp,
        score=score,
        code=code,
        blob=blob
    )
    connection.execute(clause)
