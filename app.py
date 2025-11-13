
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict

app = FastAPI(title="Evolv.AI Recommender", version="0.1.0")

# ENTRADA/SAÍDA 
class User(BaseModel):
    id: int
    nivel: str = "iniciante"              # iniciante | intermediario | avancado
    xp: int = 0
    streakDias: int = 0

class Trilha(BaseModel):
    id: int
    titulo: str
    tags: List[str] = []
    nivel: str = "iniciante"

class Conteudo(BaseModel):
    id: int
    titulo: str
    tipo: str         # "vídeo" | "artigo" | "quiz" | etc.
    duracao: int      # minutos
    nivel: str        # iniciante | intermediario | avancado
    tags: List[str] = []

class RecommendInput(BaseModel):
    user: User
    trilhas: List[Trilha] = []
    catalogo: List[Conteudo] = []
    k: int = 10

class ScoredContent(Conteudo):
    score: float

class RecommendOutput(BaseModel):
    items: List[ScoredContent]

# ======== HEURÍSTICA (MVP) =========
# Regras simples, fáceis de explicar no pitch:
# +1  se nivel do conteúdo = nivel do usuário  (ou iniciante-iniciante)
# +1  se intersecta tags da(s) trilha(s) ativa(s)
# +1  se duração <= 20 min (conteúdo curto melhora conclusão)
# +0.25 bônus se tipo diverso de teoricamente "ultimo_tipo" (placeholder)
# Ordena por score desc; empate: menor duração primeiro

def score_item(user: User, trilha_tags: set, c: Conteudo) -> float:
    score = 0.0
    if user.nivel == "iniciante" and c.nivel == "iniciante":
        score += 1.0
    elif user.nivel != "iniciante" and c.nivel != "iniciante":
        score += 1.0

    if any(t in trilha_tags for t in c.tags or []):
        score += 1.0

    if (c.duracao or 999) <= 20:
        score += 1.0
    return score

@app.get("/health")
def health():
    return {"ok": True, "service": "evolv-ai-ml"}

@app.post("/recommend", response_model=RecommendOutput)
def recommend(payload: RecommendInput):
    trilha_tags = set()
    for t in payload.trilhas:
        trilha_tags.update(t.tags or [])

    scored: List[ScoredContent] = []
    for c in payload.catalogo:
        s = score_item(payload.user, trilha_tags, c)
        scored.append(ScoredContent(**c.model_dump(), score=s))

    scored.sort(key=lambda x: (-x.score, x.duracao))
    return RecommendOutput(items=scored[: payload.k])
