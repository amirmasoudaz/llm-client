# src/db/migrate.py

"""
Clone a fixed set of tables from a SOURCE MariaDB/MySQL DB to a DESTINATION DB
with IDENTICAL schema and data (IDs, timestamps, FKs, indexes, collations).

Usage:
  export SRC_DB_HOST=127.0.0.1
  export SRC_DB_PORT=3306
  export SRC_DB_USER=src_user
  export SRC_DB_PASS=src_pass
  export SRC_DB_NAME=source_db

  export DEST_DB_HOST=127.0.0.1
  export DEST_DB_PORT=3306
  export DEST_DB_USER=dst_user
  export DEST_DB_PASS=dst_pass
  export DEST_DB_NAME=target_db

  python migrate.py
"""

import asyncio, logging
from typing import List, Tuple, Dict, Any, Sequence

from src.config import settings
import aiomysql


TABLE_CREATE_FUNDING_INSTITUTES = """
CREATE TABLE IF NOT EXISTS funding_institutes (
    id                INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    institution_name  VARCHAR(255) NOT NULL,
    department_name   VARCHAR(255) NOT NULL,
    institution_url   VARCHAR(1024) DEFAULT NULL,
    logo_address      VARCHAR(1024) DEFAULT NULL,
    city              VARCHAR(255) DEFAULT NULL,
    province          VARCHAR(255) DEFAULT NULL,
    is_active         TINYINT(1) NOT NULL DEFAULT 1,
    UNIQUE KEY uniq_inst_dept (institution_name, department_name)
) ENGINE=InnoDB AUTO_INCREMENT=1142 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

TABLE_CREATE_FUNDING_PROFESSORS = """
CREATE TABLE IF NOT EXISTS funding_professors (
    prof_hash           BINARY(32) NOT NULL,
    id                  INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    full_name           VARCHAR(512) NOT NULL,
    first_name          VARCHAR(512) NOT NULL,
    middle_name         VARCHAR(512) DEFAULT NULL,
    last_name           VARCHAR(512) NOT NULL,
    occupation          VARCHAR(512) DEFAULT NULL,
    department          VARCHAR(255) NOT NULL,
    email_address       VARCHAR(320) NOT NULL,
    url                 VARCHAR(1024) DEFAULT NULL,
    funding_institute_id INT(10) UNSIGNED NOT NULL,
    other_contact_info  JSON DEFAULT NULL,
    research_areas      JSON NOT NULL,
    credentials         TEXT DEFAULT NULL,
    area_of_expertise   JSON DEFAULT NULL,
    categories          JSON NOT NULL,
    others              JSON DEFAULT NULL,
    is_active           TINYINT(1) NOT NULL DEFAULT 1,
    UNIQUE KEY uniq_prof_hash (prof_hash),
    KEY fk_prof_inst (funding_institute_id)
) ENGINE=InnoDB AUTO_INCREMENT=46240 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

DDL_BY_TABLE: Dict[str, str] = {
    "funding_institutes": TABLE_CREATE_FUNDING_INSTITUTES,
    "funding_professors": TABLE_CREATE_FUNDING_PROFESSORS,
}

# Create/copy order (parents before children; FK table last)
TABLE_ORDER: List[str] = [
    "funding_institutes",
    "funding_professors",
]

TABLE_RENAMES: dict[str, str] = {
    "users": "students",
}

COLUMN_RENAMES = {
    "funding_requests": {"user_id": "student_id"},
}

MIGRATION_STEPS: list[tuple[str, str]] = [
    # (src_table, dest_table)
    ("funding_institutes", "funding_institutes"),
    ("funding_professors", "funding_professors"),
]

CHUNK_SIZE = settings.CLONE_CHUNK_SIZE

class DB:
    _pool: aiomysql.Pool | None = None
    _lock = asyncio.Lock()

    _log = logging.getLogger("db")

    @classmethod
    async def pool(cls) -> aiomysql.Pool:
        if cls._pool is None:
            async with cls._lock:         # double‑checked locking
                if cls._pool is None:
                    cls._pool = await aiomysql.create_pool(
                        host=settings.DB_HOST,
                        port=settings.DB_PORT,
                        user=settings.DB_USER,
                        password=settings.DB_PASS,
                        db=settings.DB_NAME,
                        minsize=settings.DB_MIN,
                        maxsize=settings.DB_MAX,
                        autocommit=True,
                        charset="utf8mb4",
                    )
                    cls._log.info("MySQL pool created")
        return cls._pool

    @classmethod
    async def execute(cls, q: str, p: Sequence[Any] = ()) -> int:
        pool = await cls.pool()
        async with pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(q, p)
            return cur.lastrowid or 0

    @classmethod
    async def fetch_one(cls, q: str, p: Sequence[Any] = ()):
        pool = await cls.pool()
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(q, p)
            return await cur.fetchone()

    @classmethod
    async def fetch_all(cls, q: str, p: Sequence[Any] = ()):
        pool = await cls.pool()
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(q, p)
            return await cur.fetchall()

    @classmethod
    async def close(cls):
        if cls._pool:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._log.info("MySQL pool closed")

class SrcDB(DB):
    _pool: aiomysql.Pool | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def pool(cls) -> aiomysql.Pool:
        if cls._pool is None:
            async with cls._lock:
                if cls._pool is None:
                    cls._pool = await aiomysql.create_pool(
                        host=settings.SRC_DB_HOST,
                        port=settings.SRC_DB_PORT,
                        user=settings.SRC_DB_USER,
                        password=settings.SRC_DB_PASS,
                        db=settings.SRC_DB_NAME,
                        minsize=settings.DB_MIN,
                        maxsize=settings.DB_MAX,
                        autocommit=True,
                        charset="utf8mb4",
                    )
                    cls._log.info("Source MySQL pool created")
        return cls._pool


class DestDB(DB):
    _pool: aiomysql.Pool | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def pool(cls) -> aiomysql.Pool:
        if cls._pool is None:
            async with cls._lock:
                if cls._pool is None:
                    cls._pool = await aiomysql.create_pool(
                        host=settings.DEST_DB_HOST,
                        port=settings.DEST_DB_PORT,
                        user=settings.DEST_DB_USER,
                        password=settings.DEST_DB_PASS,
                        db=settings.DEST_DB_NAME,
                        minsize=settings.DB_MIN,
                        maxsize=settings.DB_MAX,
                        autocommit=True,
                        charset="utf8mb4",
                    )
                    cls._log.info("Destination MySQL pool created")
        return cls._pool

async def _ensure_students_exists():
    row = await DestDB.fetch_one(
        "SELECT COUNT(*) AS c FROM information_schema.tables "
        "WHERE table_schema=%s AND table_name='students'",
        (settings.DEST_DB_NAME,),
    )
    if not row or not row["c"]:
        raise RuntimeError("DEST is missing required table `students`")

async def _verify_student_references():
    miss_fr = await DestDB.fetch_one("""
        SELECT COUNT(*) AS missing
        FROM (SELECT DISTINCT student_id FROM funding_requests) r
        LEFT JOIN students s ON s.id = r.student_id
        WHERE s.id IS NULL
    """)
    miss_fc = await DestDB.fetch_one("""
        SELECT COUNT(*) AS missing
        FROM (SELECT DISTINCT user_id FROM funding_credentials) c
        LEFT JOIN students s ON s.id = c.user_id
        WHERE s.id IS NULL
    """)
    if miss_fr["missing"] or miss_fc["missing"]:
        raise AssertionError(
            f"Orphan references: funding_requests={miss_fr['missing']}, funding_credentials={miss_fc['missing']}"
        )

async def _exec_many(conn: aiomysql.Connection, sql: str, rows: List[Tuple[Any, ...]]):
    async with conn.cursor() as cur:
        await cur.executemany(sql, rows)

async def truncate_destination_tables(reset_ai: bool = True):
    dpool = await DestDB.pool()
    async with dpool.acquire() as dconn, dconn.cursor() as dcur:
        await dcur.execute("SET FOREIGN_KEY_CHECKS=0")
        for t in reversed(TABLE_ORDER):
            try:
                await dcur.execute(f"TRUNCATE TABLE `{t}`")
            except Exception:
                await dcur.execute(f"DELETE FROM `{t}`")
                if reset_ai:
                    try:
                        await dcur.execute(f"ALTER TABLE `{t}` AUTO_INCREMENT = 1")
                    except Exception:
                        pass
        await dcur.execute("SET FOREIGN_KEY_CHECKS=1")

async def _get_dest_insertable_columns(table: str) -> set[str]:
    rows = await DestDB.fetch_all(
        """
        SELECT COLUMN_NAME, EXTRA, GENERATION_EXPRESSION
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s
        """,
        (settings.DEST_DB_NAME, table),
    )
    insertable = set()
    for r in rows:
        extra = (r.get("EXTRA") or "").upper()
        gen_expr = r.get("GENERATION_EXPRESSION")
        if "GENERATED" in extra or (gen_expr is not None and gen_expr != ""):
            continue
        insertable.add(r["COLUMN_NAME"])
    return insertable


async def _get_auto_increment(db: type[DB], table: str, dbname: Optional[str]) -> int | None:
    if not dbname:
        return None
    row = await db.fetch_one(
        """
        SELECT AUTO_INCREMENT
        FROM information_schema.tables
        WHERE table_schema=%s AND table_name=%s
        """,
        (dbname, table),
    )
    return int(row["AUTO_INCREMENT"]) if row and row["AUTO_INCREMENT"] is not None else None

async def _get_columns_for(db: type[DB], dbname: Optional[str], table: str) -> list[str]:
    if not dbname:
        return []
    rows = await db.fetch_all(
        """
        SELECT COLUMN_NAME
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s
        ORDER BY ORDINAL_POSITION
        """,
        (dbname, table),
    )
    return [r["COLUMN_NAME"] for r in rows]

async def copy_table_mapped(src_table: str, dest_table: str):
    src_cols = await _get_columns_for(SrcDB, settings.SRC_DB_NAME, src_table)
    dest_cols = await _get_dest_insertable_columns(dest_table)

    colmap = COLUMN_RENAMES.get(src_table, {})
    select_items, dest_list = [], []

    for s in src_cols:
        d = colmap.get(s, s)
        if d in dest_cols:
            select_items.append(f"`{s}` AS `{d}`")
            dest_list.append(f"`{d}`")

    if not dest_list:
        raise RuntimeError(f"No matching columns between src `{src_table}` and dest `{dest_table}`")

    select_sql = f"SELECT {', '.join(select_items)} FROM `{src_table}`"
    insert_sql = f"INSERT INTO `{dest_table}` ({', '.join(dest_list)}) VALUES ({', '.join(['%s'] * len(dest_list))})"

    spool = await SrcDB.pool()
    dpool = await DestDB.pool()

    async with dpool.acquire() as dconn:
        async with dconn.cursor() as dcur:
            await dcur.execute("SET FOREIGN_KEY_CHECKS=0")

        async with (spool.acquire() as sconn):
            async with sconn.cursor(aiomysql.SSCursor) as scur:
                await scur.execute(select_sql)
                batch: list[tuple[Any, ...]] = []
                while True:
                    row = await scur.fetchone()
                    if row is None:
                        if batch:
                            await _exec_many(dconn, insert_sql, batch)
                            batch.clear()
                        break
                    # row is already ordered to match select_items
                    batch.append(tuple(row))
                    if len(batch) >= CHUNK_SIZE:
                        await _exec_many(dconn, insert_sql, batch)
                        batch.clear()

        async with dconn.cursor() as dcur:
            await dcur.execute("SET FOREIGN_KEY_CHECKS=1")

    src_cnt = await SrcDB.fetch_one(f"SELECT COUNT(*) AS c FROM `{src_table}`")
    dst_cnt = await DestDB.fetch_one(f"SELECT COUNT(*) AS c FROM `{dest_table}`")
    if (src_cnt or {}).get("c") != (dst_cnt or {}).get("c"):
        raise AssertionError(f"Row count mismatch src `{src_table}` vs dest `{dest_table}`: {src_cnt['c']} != {dst_cnt['c']}")

    src_auto = await _get_auto_increment(SrcDB, src_table, settings.SRC_DB_NAME)
    if src_auto:
        await DestDB.execute(f"ALTER TABLE `{dest_table}` AUTO_INCREMENT = {int(src_auto)}")

async def clone_all():
    await _ensure_students_exists()

    print("Truncating destination tables …")
    await truncate_destination_tables()

    print("Copying data with mappings …")
    for src_table, default_dest in MIGRATION_STEPS:
        dest_table = TABLE_RENAMES.get(src_table, default_dest)
        print(f"[{src_table} -> {dest_table}] Copy start")
        await copy_table_mapped(src_table, dest_table)
        print(f"[{src_table} -> {dest_table}] Copy OK")

    await _verify_student_references()
    print("All tables copied and verified ✅")

async def main():
    required = [
        settings.SRC_DB_HOST, settings.SRC_DB_USER, settings.SRC_DB_NAME,
        settings.DEST_DB_HOST, settings.DEST_DB_USER, settings.DEST_DB_NAME
    ]
    if any(v is None for v in required):
        raise RuntimeError(f"Missing required database settings for migration")

    try:
        await clone_all()
    finally:
        # Cleanly close pools
        await SrcDB.close()
        await DestDB.close()

if __name__ == "__main__":
    asyncio.run(main())
