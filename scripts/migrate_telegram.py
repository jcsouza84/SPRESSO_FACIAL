"""
Migração: adiciona coluna telegram_sent em alert_records.
Seguro para rodar múltiplas vezes (verifica existência antes de alterar).

Uso:
    python scripts/migrate_telegram.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "spresso.db"


def column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def run() -> None:
    if not DB_PATH.exists():
        print(f"Banco não encontrado em {DB_PATH}. Nenhuma migração necessária.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    try:
        if not column_exists(cur, "alert_records", "telegram_sent"):
            cur.execute("ALTER TABLE alert_records ADD COLUMN telegram_sent BOOLEAN DEFAULT 0")
            conn.commit()
            print("Migração concluída. Coluna 'telegram_sent' adicionada em alert_records.")
        else:
            print("Nenhuma migração necessária — coluna 'telegram_sent' já existe.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
