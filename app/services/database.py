from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


Base = declarative_base()


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    category = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@dataclass
class QueryFilter:
    category: Optional[str] = None
    name_contains: Optional[str] = None
    description_contains: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class QueryResult:
    products: List[Dict[str, Any]]
    total_count: int
    query_summary: str


class DatabaseService:
    """Fast SQLite-based database service for structured queries."""

    def __init__(self, db_path: str = "data.sqlite"):
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def load_csv_data(self, csv_path: Path) -> None:
        """Load CSV data into the database, replacing existing data."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Clear existing data
        self.session.query(Product).delete()

        # Insert new data
        for _, row in df.iterrows():
            product = Product(
                name=str(row['name']),
                description=str(row['description']),
                category=str(row['category'])
            )
            self.session.add(product)

        self.session.commit()
        print(f"✅ Loaded {len(df)} products from {csv_path}")

    def query_products(self, filters: QueryFilter) -> QueryResult:
        """Execute structured query with filters."""
        # Build base query
        query = self.session.query(Product)
        count_query = self.session.query(Product)

        # Apply filters
        conditions = []
        if filters.category:
            query = query.filter(Product.category.ilike(f"%{filters.category}%"))
            count_query = count_query.filter(Product.category.ilike(f"%{filters.category}%"))
            conditions.append(f"category contains '{filters.category}'")

        if filters.name_contains:
            query = query.filter(Product.name.ilike(f"%{filters.name_contains}%"))
            count_query = count_query.filter(Product.name.ilike(f"%{filters.name_contains}%"))
            conditions.append(f"name contains '{filters.name_contains}'")

        if filters.description_contains:
            query = query.filter(Product.description.ilike(f"%{filters.description_contains}%"))
            count_query = count_query.filter(Product.description.ilike(f"%{filters.description_contains}%"))
            conditions.append(f"description contains '{filters.description_contains}'")

        # Get total count before pagination
        total_count = count_query.count()

        # Apply pagination
        query = query.offset(filters.offset).limit(filters.limit)

        # Execute query
        products = query.all()

        # Convert to dictionaries
        product_dicts = [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "category": p.category,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            }
            for p in products
        ]

        # Generate query summary
        if conditions:
            query_summary = f"Found {total_count} products where " + " and ".join(conditions)
        else:
            query_summary = f"Found {total_count} products (showing all)"

        if filters.limit < total_count:
            query_summary += f" (showing {len(product_dicts)} of {total_count})"

        return QueryResult(
            products=product_dicts,
            total_count=total_count,
            query_summary=query_summary
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_products = self.session.query(Product).count()
        categories = self.session.query(Product.category).distinct().all()

        return {
            "total_products": total_products,
            "categories": [cat[0] for cat in categories],
            "database_path": str(self.db_path)
        }

    def close(self):
        """Close database connection."""
        self.session.close()


__all__ = ["DatabaseService", "QueryFilter", "QueryResult", "Product"]