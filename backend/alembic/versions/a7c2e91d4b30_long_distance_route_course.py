"""long distance route course

Revision ID: a7c2e91d4b30
Revises: 7b3e91c4d2a6
Create Date: 2026-11-11 15:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = 'a7c2e91d4b30'
down_revision: str | None = '7b3e91c4d2a6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'long_distance_routes',
        sa.Column('course', sa.String(length=300), server_default='', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('long_distance_routes', 'course')
