"""user gpx mode

Revision ID: 7b3e91c4d2a6
Revises: 20177c673345
Create Date: 2026-10-26 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '7b3e91c4d2a6'
down_revision: str | None = '20177c673345'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('users', sa.Column('gpx_mode', sa.String(length=20), server_default='combined', nullable=False))


def downgrade() -> None:
    op.drop_column('users', 'gpx_mode')
