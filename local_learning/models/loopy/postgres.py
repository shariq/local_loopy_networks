import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, Float, PickleType, DateTime
from sqlalchemy.dialects import postgresql
import os
import datetime

import logging

connection_metas = {}

def initialize(postgres_uri):
    connection = sqlalchemy.create_engine(postgres_uri, client_encoding='utf8')
    meta = sqlalchemy.MetaData(bind=connection, reflect=True)

    if 'results' not in meta.tables:
        results = Table('results', meta,
            Column('timestamp', DateTime),
            Column('score', Float),
            Column('code', String),
            Column('blob', PickleType)
        )

        # Create the above tables
        meta.create_all(connection)

    connection_metas[postgres_uri] = (connection, meta)

    return connection, meta

def save(postgres_uri, score, code, blob):
    if postgres_uri not in connection_metas:
        connection, meta = initialize(postgres_uri)
    connection, meta = connection_metas[postgres_uri]
    timestamp = datetime.datetime.now()
    clause = meta.tables['results'].insert().values(
        timestamp=timestamp,
        score=score,
        code=code,
        blob=blob
    )
    try:
        connection.execute(clause)
    except Exception as e:
        initialize(postgres_uri)
        logger.error(e, exc_info=True)
