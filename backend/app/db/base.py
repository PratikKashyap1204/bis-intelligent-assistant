"""SQLAlchemy declarative base and reusable timestamp mixins."""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Declarative base for every ORM model in this project.

    All model classes inherit from Base so that Base.metadata.create_all()
    can discover and create every table in one call.
    """


class CreatedAtMixin:
    """
    Mixin that adds a single ``created_at`` timestamp column.

    Use for entities that are inserted once and never logically 'updated'
    (e.g. Clause, Laboratory, Document).
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class TimestampMixin(CreatedAtMixin):
    """
    Mixin that adds both ``created_at`` and ``updated_at`` columns.

    ``onupdate=func.now()`` ensures SQLAlchemy sets the value automatically
    whenever the ORM issues an UPDATE for this row.

    Use for entities that change over time (e.g. Standard, which gets revised).
    """

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
