from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


Base = declarative_base()


class BrainSpecimen(Base):
    __tablename__ = "brain_specimens"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Basic identifiers
    subject_id = Column(String, nullable=False, index=True)
    repository = Column(String, nullable=True)

    # Subject demographics
    race = Column(String, nullable=True, index=True)
    subject_sex = Column(String, nullable=True, index=True)
    subject_age = Column(Integer, nullable=True, index=True)
    ethnicity = Column(String, nullable=True)

    # Clinical information
    neuropathology_diagnosis = Column(Text, nullable=True)
    clinical_brain_diagnosis = Column(Text, nullable=True)
    genetic_diagnosis = Column(String, nullable=True)
    non_brain_diagnosis = Column(Text, nullable=True)
    manner_of_death = Column(String, nullable=True, index=True)

    # Tissue information
    brain_hemisphere = Column(Text, nullable=True)
    brain_region = Column(Text, nullable=True)
    tissue_source = Column(String, nullable=True)
    pmi_hours = Column(Float, nullable=True)  # Post-mortem interval
    rin = Column(Float, nullable=True)  # RNA integrity number
    preparation = Column(String, nullable=True)

    # Research flags
    non_diagnostic_flag = Column(String, nullable=True)
    pre_mortem = Column(String, nullable=True)

    # Pathology scores (many are "No Results Reported" in sample)
    thal_phase = Column(String, nullable=True)
    braak_nft_stage = Column(String, nullable=True)
    cerad_score = Column(String, nullable=True)
    a_score = Column(String, nullable=True)
    b_score = Column(String, nullable=True)
    c_score = Column(String, nullable=True)
    adnc = Column(String, nullable=True)  # Alzheimer disease neuropathologic change
    lewy_pathology = Column(String, nullable=True)

    # ICD codes
    icd_clinical_brain = Column(String, nullable=True)
    icd_genetic = Column(String, nullable=True)
    icd_neuropathology = Column(String, nullable=True)
    icd_non_brain = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@dataclass
class BrainQueryFilter:
    # Identifiers
    subject_id: Optional[str] = None  # Exact or partial subject ID match

    # Demographics
    race: Optional[str] = None
    subject_sex: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    ethnicity: Optional[str] = None

    # Clinical
    manner_of_death: Optional[str] = None
    diagnosis_contains: Optional[str] = None
    repository: Optional[str] = None

    # Tissue quality
    pmi_max: Optional[float] = None  # Maximum post-mortem interval
    rin_min: Optional[float] = None  # Minimum RNA integrity

    # Brain regions
    brain_region_contains: Optional[str] = None
    hemisphere: Optional[str] = None

    # Pagination
    limit: int = 100
    offset: int = 0


@dataclass
class BrainQueryResult:
    specimens: List[Dict[str, Any]]
    total_count: int
    query_summary: str


class BrainDatabaseService:
    """Fast SQLite-based database service for brain research data."""

    def __init__(self, db_path: str = "brain_data.sqlite"):
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def load_csv_data(self, csv_path: Path) -> None:
        """Load brain research CSV data into the database."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"🧠 Loading brain research data from {csv_path}...")

        # Read CSV with pandas
        df = pd.read_csv(csv_path)
        print(f"📊 Found {len(df)} specimens with {len(df.columns)} columns")

        # Clear existing data
        self.session.query(BrainSpecimen).delete()

        # Process and insert data
        inserted_count = 0
        for _, row in df.iterrows():
            try:
                # Parse age - handle non-numeric values
                age = None
                if pd.notna(row.get('Subject Age')) and str(row.get('Subject Age')).isdigit():
                    age = int(row.get('Subject Age'))

                # Parse PMI - handle non-numeric values
                pmi = None
                if pd.notna(row.get('PMI (hours)')):
                    try:
                        pmi = float(str(row.get('PMI (hours)')).replace(',', ''))
                    except (ValueError, TypeError):
                        pmi = None

                # Parse RIN - handle non-numeric values
                rin = None
                if pd.notna(row.get('RIN')):
                    try:
                        rin_val = str(row.get('RIN'))
                        if rin_val != "99.99":  # Seems to be a placeholder
                            rin = float(rin_val)
                    except (ValueError, TypeError):
                        rin = None

                specimen = BrainSpecimen(
                    subject_id=str(row.get('Subject ID', '')),
                    repository=str(row.get('Repository', '')) if pd.notna(row.get('Repository')) else None,

                    # Demographics
                    race=str(row.get('Race', '')) if pd.notna(row.get('Race')) else None,
                    subject_sex=str(row.get('Subject Sex', '')) if pd.notna(row.get('Subject Sex')) else None,
                    subject_age=age,
                    ethnicity=str(row.get('Ethnicity', '')) if pd.notna(row.get('Ethnicity')) else None,

                    # Clinical
                    neuropathology_diagnosis=str(row.get('Neuropathology Diagnosis', '')) if pd.notna(row.get('Neuropathology Diagnosis')) else None,
                    clinical_brain_diagnosis=str(row.get('Clinical Brain Diagnosis (Basis for Clinical Diagnosis)', '')) if pd.notna(row.get('Clinical Brain Diagnosis (Basis for Clinical Diagnosis)')) else None,
                    genetic_diagnosis=str(row.get('Genetic Diagnosis', '')) if pd.notna(row.get('Genetic Diagnosis')) else None,
                    non_brain_diagnosis=str(row.get('Non Brain Diagnosis', '')) if pd.notna(row.get('Non Brain Diagnosis')) else None,
                    manner_of_death=str(row.get('Manner of Death', '')) if pd.notna(row.get('Manner of Death')) else None,

                    # Tissue information
                    brain_hemisphere=str(row.get('Brain Hemisphere', '')) if pd.notna(row.get('Brain Hemisphere')) else None,
                    brain_region=str(row.get('Brain Region', '')) if pd.notna(row.get('Brain Region')) else None,
                    tissue_source=str(row.get('Tissue Source', '')) if pd.notna(row.get('Tissue Source')) else None,
                    pmi_hours=pmi,
                    rin=rin,
                    preparation=str(row.get('Preparation', '')) if pd.notna(row.get('Preparation')) else None,

                    # Research flags
                    non_diagnostic_flag=str(row.get('Non-Diagnostic Flag', '')) if pd.notna(row.get('Non-Diagnostic Flag')) else None,
                    pre_mortem=str(row.get('Pre-Mortem', '')) if pd.notna(row.get('Pre-Mortem')) else None,

                    # Pathology scores
                    thal_phase=str(row.get('Thal Phase', '')) if pd.notna(row.get('Thal Phase')) else None,
                    braak_nft_stage=str(row.get('Braak NFT Stage', '')) if pd.notna(row.get('Braak NFT Stage')) else None,
                    cerad_score=str(row.get('CERAD Score', '')) if pd.notna(row.get('CERAD Score')) else None,
                    a_score=str(row.get('A Score', '')) if pd.notna(row.get('A Score')) else None,
                    b_score=str(row.get('B Score', '')) if pd.notna(row.get('B Score')) else None,
                    c_score=str(row.get('C Score', '')) if pd.notna(row.get('C Score')) else None,
                    adnc=str(row.get('ADNC', '')) if pd.notna(row.get('ADNC')) else None,
                    lewy_pathology=str(row.get('Lewy Pathology', '')) if pd.notna(row.get('Lewy Pathology')) else None,

                    # ICD codes
                    icd_clinical_brain=str(row.get('ICD for Clinical Brain Diagnosis', '')) if pd.notna(row.get('ICD for Clinical Brain Diagnosis')) else None,
                    icd_genetic=str(row.get('ICD for Genetic Diagnosis', '')) if pd.notna(row.get('ICD for Genetic Diagnosis')) else None,
                    icd_neuropathology=str(row.get('ICD for Neuropathology Diagnosis', '')) if pd.notna(row.get('ICD for Neuropathology Diagnosis')) else None,
                    icd_non_brain=str(row.get('ICD for Non Brain Diagnosis', '')) if pd.notna(row.get('ICD for Non Brain Diagnosis')) else None,
                )

                self.session.add(specimen)
                inserted_count += 1

                # Commit in batches for better performance
                if inserted_count % 1000 == 0:
                    self.session.commit()
                    print(f"  📥 Inserted {inserted_count} specimens...")

            except Exception as e:
                print(f"⚠️  Error processing row {row.get('Subject ID', 'unknown')}: {e}")
                continue

        self.session.commit()
        print(f"✅ Successfully loaded {inserted_count} brain specimens into database")

    def query_specimens(self, filters: BrainQueryFilter) -> BrainQueryResult:
        """Execute structured query with filters for brain specimens."""
        # Build base query
        query = self.session.query(BrainSpecimen)
        count_query = self.session.query(BrainSpecimen)

        # Apply filters
        conditions = []

        if filters.subject_id:
            query = query.filter(BrainSpecimen.subject_id.ilike(f"%{filters.subject_id}%"))
            count_query = count_query.filter(BrainSpecimen.subject_id.ilike(f"%{filters.subject_id}%"))
            conditions.append(f"subject ID contains '{filters.subject_id}'")

        if filters.race:
            query = query.filter(BrainSpecimen.race.ilike(f"%{filters.race}%"))
            count_query = count_query.filter(BrainSpecimen.race.ilike(f"%{filters.race}%"))
            conditions.append(f"race contains '{filters.race}'")

        if filters.subject_sex:
            query = query.filter(BrainSpecimen.subject_sex.ilike(f"%{filters.subject_sex}%"))
            count_query = count_query.filter(BrainSpecimen.subject_sex.ilike(f"%{filters.subject_sex}%"))
            conditions.append(f"sex is '{filters.subject_sex}'")

        if filters.age_min is not None:
            query = query.filter(BrainSpecimen.subject_age >= filters.age_min)
            count_query = count_query.filter(BrainSpecimen.subject_age >= filters.age_min)
            conditions.append(f"age >= {filters.age_min}")

        if filters.age_max is not None:
            query = query.filter(BrainSpecimen.subject_age <= filters.age_max)
            count_query = count_query.filter(BrainSpecimen.subject_age <= filters.age_max)
            conditions.append(f"age <= {filters.age_max}")

        if filters.manner_of_death:
            query = query.filter(BrainSpecimen.manner_of_death.ilike(f"%{filters.manner_of_death}%"))
            count_query = count_query.filter(BrainSpecimen.manner_of_death.ilike(f"%{filters.manner_of_death}%"))
            conditions.append(f"manner of death contains '{filters.manner_of_death}'")

        if filters.diagnosis_contains:
            diagnosis_filter = (
                BrainSpecimen.clinical_brain_diagnosis.ilike(f"%{filters.diagnosis_contains}%") |
                BrainSpecimen.neuropathology_diagnosis.ilike(f"%{filters.diagnosis_contains}%")
            )
            query = query.filter(diagnosis_filter)
            count_query = count_query.filter(diagnosis_filter)
            conditions.append(f"diagnosis contains '{filters.diagnosis_contains}'")

        if filters.brain_region_contains:
            query = query.filter(BrainSpecimen.brain_region.ilike(f"%{filters.brain_region_contains}%"))
            count_query = count_query.filter(BrainSpecimen.brain_region.ilike(f"%{filters.brain_region_contains}%"))
            conditions.append(f"brain region contains '{filters.brain_region_contains}'")

        if filters.repository:
            query = query.filter(BrainSpecimen.repository.ilike(f"%{filters.repository}%"))
            count_query = count_query.filter(BrainSpecimen.repository.ilike(f"%{filters.repository}%"))
            conditions.append(f"repository contains '{filters.repository}'")

        if filters.pmi_max is not None:
            query = query.filter(BrainSpecimen.pmi_hours <= filters.pmi_max)
            count_query = count_query.filter(BrainSpecimen.pmi_hours <= filters.pmi_max)
            conditions.append(f"PMI <= {filters.pmi_max} hours")

        if filters.rin_min is not None:
            query = query.filter(BrainSpecimen.rin >= filters.rin_min)
            count_query = count_query.filter(BrainSpecimen.rin >= filters.rin_min)
            conditions.append(f"RIN >= {filters.rin_min}")

        # Get total count before pagination
        total_count = count_query.count()

        # Apply pagination
        query = query.offset(filters.offset).limit(filters.limit)

        # Execute query
        specimens = query.all()

        # Convert to dictionaries
        specimen_dicts = [
            {
                "id": s.id,
                "subject_id": s.subject_id,
                "repository": s.repository,
                "race": s.race,
                "subject_sex": s.subject_sex,
                "subject_age": s.subject_age,
                "ethnicity": s.ethnicity,
                "neuropathology_diagnosis": s.neuropathology_diagnosis,
                "clinical_brain_diagnosis": s.clinical_brain_diagnosis,
                "manner_of_death": s.manner_of_death,
                "brain_region": s.brain_region,
                "pmi_hours": s.pmi_hours,
                "rin": s.rin,
                "preparation": s.preparation,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in specimens
        ]

        # Generate query summary
        if conditions:
            query_summary = f"Found {total_count} brain specimens where " + " and ".join(conditions)
        else:
            query_summary = f"Found {total_count} brain specimens (showing all)"

        if filters.limit < total_count:
            query_summary += f" (showing {len(specimen_dicts)} of {total_count})"

        return BrainQueryResult(
            specimens=specimen_dicts,
            total_count=total_count,
            query_summary=query_summary
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics for brain specimens."""
        total_specimens = self.session.query(BrainSpecimen).count()

        # Get unique values for categorical fields
        races = [r[0] for r in self.session.query(BrainSpecimen.race).distinct().all() if r[0]]
        sexes = [s[0] for s in self.session.query(BrainSpecimen.subject_sex).distinct().all() if s[0]]
        repositories = [r[0] for r in self.session.query(BrainSpecimen.repository).distinct().all() if r[0]]

        # Age statistics
        avg_age = self.session.execute(
            text("SELECT AVG(subject_age) FROM brain_specimens WHERE subject_age IS NOT NULL")
        ).scalar()

        return {
            "total_specimens": total_specimens,
            "races": races,
            "sexes": sexes,
            "repositories": repositories,
            "average_age": round(avg_age, 1) if avg_age else None,
            "database_path": str(self.db_path)
        }

    def close(self):
        """Close database connection."""
        self.session.close()


__all__ = ["BrainDatabaseService", "BrainQueryFilter", "BrainQueryResult", "BrainSpecimen"]