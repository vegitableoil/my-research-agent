"""SQLAlchemy ORM schema for the findings and results database."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[str] = mapped_column(Text)
    year: Mapped[int] = mapped_column(Integer)
    venue: Mapped[str] = mapped_column(String)
    source_url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="pending")   # pending | approved | rejected
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    summary: Mapped["PaperSummary | None"] = relationship(back_populates="paper")


class PaperSummary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    novelty: Mapped[str] = mapped_column(Text)
    system_design: Mapped[str] = mapped_column(Text)
    datasets_used: Mapped[str] = mapped_column(Text)
    architecture_summary: Mapped[str] = mapped_column(Text)
    performance_metrics: Mapped[dict] = mapped_column(JSON)
    limitations: Mapped[str] = mapped_column(Text)
    implications: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    paper: Mapped["Paper"] = relationship(back_populates="summary")


class BaselineModel(Base):
    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    model_type: Mapped[str] = mapped_column(String)       # stdlib | paper_found | paper_replicated
    source_paper_id: Mapped[str | None] = mapped_column(ForeignKey("papers.id"), nullable=True)
    github_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, unique=True)
    dataset_name: Mapped[str] = mapped_column(String)
    proposed_model_artifact: Mapped[str] = mapped_column(Text)
    hpo_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    results: Mapped[list["ModelResult"]] = relationship(back_populates="experiment")


class ModelResult(Base):
    __tablename__ = "model_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    model_name: Mapped[str] = mapped_column(String)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    detection_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_alarm_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    pr_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_size_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    checkpoint_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship(back_populates="results")
