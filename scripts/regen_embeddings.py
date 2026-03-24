"""
Script de manutenção: regenera embeddings de todas as fotos com embedding NULL.
Executar uma única vez após correção do bug de geração de embeddings.

Uso: .venv/bin/python scripts/regen_embeddings.py
"""
import sys
import sqlite3
from pathlib import Path

# Garante que o path do projeto está disponível
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recognition.embeddings import get_embedding_from_file

DB_PATH = Path("data/spresso.db")

def main():
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        "SELECT id, person_id, path FROM person_photos WHERE embedding IS NULL"
    ).fetchall()

    print(f"Fotos sem embedding: {len(rows)}")
    if not rows:
        print("Nada a fazer.")
        conn.close()
        return

    ok = 0
    skip = 0
    for photo_id, person_id, path in rows:
        p = Path(path)
        if not p.exists():
            print(f"  [SKIP] photo_id={photo_id} arquivo não encontrado: {path}")
            skip += 1
            continue

        emb = get_embedding_from_file(p)
        if emb is not None:
            conn.execute(
                "UPDATE person_photos SET embedding=? WHERE id=?",
                (emb.tobytes(), photo_id)
            )
            print(f"  [OK]   photo_id={photo_id} person_id={person_id} — embedding {emb.shape} gerado")
            ok += 1
        else:
            print(f"  [FAIL] photo_id={photo_id} — rosto não detectável na imagem")
            skip += 1

    conn.commit()
    conn.close()
    print(f"\nConcluído: {ok} gerado(s), {skip} ignorado(s).")

if __name__ == "__main__":
    main()
