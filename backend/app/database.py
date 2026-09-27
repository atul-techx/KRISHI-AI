import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings

# For async operations
engine = create_async_engine(settings.DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

import os

def use_sqlite_fallback():
    global engine, AsyncSessionLocal
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_file = os.path.join(base_dir, "krishi.db").replace("\\", "/")
    sqlite_url = f"sqlite+aiosqlite:///{db_file}"
    print(f"WARNING: Database connection failed. Switching to local SQLite database: {sqlite_url}", file=sys.stderr, flush=True)
    engine = create_async_engine(sqlite_url, echo=False)
    AsyncSessionLocal.configure(bind=engine)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
