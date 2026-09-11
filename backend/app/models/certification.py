"""CertificationRequirement model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from sqlalchemy import BigInteger, Boolean, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.product import Product
    from app.models.standard import Standard


class CertificationRequirement(Base):
    """
    Certification requirement linking a product to a BIS standard.

    A single product may have multiple certification requirements
    (e.g. BIS mark under IS 302, QCO under a separate gazette notification).
    """

    __tablename__ = "certification_requirements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    standard_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("standards.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # e.g. "BIS_CERTIFICATION_MARK", "SELF_DECLARATION", "COMPULSORY_REGISTRATION"
    scheme: Mapped[Optional[str]] = mapped_column(String(100))

    # True if certification is legally mandatory (via QCO or statute)
    mandatory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # True if this requirement is backed by a Quality Control Order
    qco: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Flexible JSON field for detailed requirement text / checklist
    requirements: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    # URL to the gazette / BIS page describing this requirement
    source_url: Mapped[Optional[str]] = mapped_column(String(2000))

    # relationships
    product: Mapped[Product] = relationship(back_populates="certification_requirements")
    standard: Mapped[Standard] = relationship(back_populates="certification_requirements")
