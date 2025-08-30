from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./ai_api.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class APIRequest(Base):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(String)
    response_data = Column(String)

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()