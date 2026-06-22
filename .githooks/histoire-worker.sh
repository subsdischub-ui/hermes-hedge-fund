#!/usr/bin/env bash
# Narrative auto-documentation worker ("Histoire du code").
#
# Launched DETACHED by .githooks/post-commit. Summarises the commits since the
# last recorded point into a CONCISE, BOUNDED running narrative and injects it
# into a marker-delimited section of ETAT_DES_LIEUX.md. A durable copy is kept
# inside .git/ so the story survives even if ETAT_DES_LIEUX.md is regenerated.
# Never blocks or breaks a commit (always exits 0; runs in the background).
set -uo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$root" || exit 0
gitdir="$(git rev-parse --git-dir 2>/dev/null)" || exit 0
hookdir="$(cd "$(dirname "$0")" && pwd)"

DOC="ETAT_DES_LIEUX.md"
BEGIN="<!-- HISTOIRE-DU-CODE:DEBUT (auto — ne pas éditer à la main) -->"
END="<!-- HISTOIRE-DU-CODE:FIN -->"
HEADING="## 🧬 Histoire du code (récit condensé, auto-généré après chaque commit)"
# Runtime state lives in .git/ (per-clone, never committed):
BACKUP="$gitdir/hermes-histoire-du-code.md"
LASTREF="$gitdir/histoire.lastref"
LOCK="$gitdir/histoire.lock"
LOG="$gitdir/histoire.log"
MAXLINES=70                          # hard cap so the section never grows long

log(){ printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG" 2>/dev/null || true; }

# Generate via Hermès's OWN LLM router (DeepSeek primary + local llama.cpp
# fallback) through the project virtualenv, so the hook reuses the project's
# working credentials/config — no Claude keychain/login needed.
PY_BIN="$root/.venv/bin/python"
[ -x "$PY_BIN" ] || PY_BIN="$(command -v python3 || true)"
LLM_SCRIPT="$hookdir/histoire-llm.py"
{ [ -n "$PY_BIN" ] && [ -f "$LLM_SCRIPT" ]; } || { log "python/llm script missing; skip"; exit 0; }

# --- single-flight lock (steal if stale > 10 min) ---
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -prune -mmin +10 2>/dev/null)" ]; then
  rmdir "$LOCK" 2>/dev/null || true
fi
mkdir "$LOCK" 2>/dev/null || { log "busy; skip"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# --- prior narrative: prefer the DOC marker block, else the durable .git copy ---
prior=""
if [ -f "$DOC" ] && grep -qF "$BEGIN" "$DOC" 2>/dev/null; then
  prior="$(awk -v b="$BEGIN" -v e="$END" 'index($0,b){f=1;next} index($0,e){f=0} f' "$DOC")"
elif [ -f "$BACKUP" ]; then
  prior="$(cat "$BACKUP" 2>/dev/null || true)"
fi

# --- which commits to integrate ---
from="$(cat "$LASTREF" 2>/dev/null || true)"
if [ -z "${prior// }" ] && [ ! -f "$BACKUP" ]; then
  # First run ever: seed from the FULL history (bounded) to tell the whole story.
  commits="$(git log --no-merges --reverse --format='- %cs `%h` %s' 2>/dev/null | tail -n 300)"
elif [ -n "$from" ] && git cat-file -e "${from}^{commit}" 2>/dev/null; then
  commits="$(git log --no-merges --reverse --format='- %cs `%h` %s' "${from}..HEAD" 2>/dev/null | head -n 60)"
else
  commits="$(git log --no-merges -1 --reverse --format='- %cs `%h` %s' HEAD 2>/dev/null)"
fi
if [ -z "${commits// }" ]; then log "no new commits; skip"; exit 0; fi

# --- build prompt (static rules verbatim in a quoted heredoc; data appended) ---
rules="$(cat <<'RULES'
Tu maintiens un RÉCIT condensé et DURABLE de l'évolution du code de « Hermès » (agent de hedge-fund crypto autonome).
On te fournit le RÉCIT ACTUEL (peut être vide) puis les NOUVEAUX commits à intégrer (ordre ancien → récent).
Renvoie UNIQUEMENT le récit MIS À JOUR — en français, en Markdown, sans préambule, sans bloc de code, et SANS titre de section (le titre est géré ailleurs).

Règles :
- Si le RÉCIT ACTUEL est vide, construis-le à partir des commits fournis (qui peuvent représenter tout l'historique), en condensant fortement le passé.
- Intègre chaque nouveau commit comme UNE puce brève (≤ 25 mots), préfixée par sa date et son hash court entre backticks (ex. « - 2026-06-22 `a1b2c3d` … »).
- L'ENSEMBLE doit rester COURT : 70 lignes MAXIMUM. Le document ne doit jamais devenir long.
- Pour tenir dans ce budget : FUSIONNE/condense les entrées ANCIENNES en résumés thématiques très courts (par thème ou par période). Regrouper n'est pas effacer — le récit doit toujours raconter, même en une ligne, tout ce qui a été fait depuis le début.
- Garde les ~10 commits les plus récents en puces individuelles ; résume de plus en plus les plus anciens.
- Utilise deux intertitres en gras : « **Fondations & passé (résumé)** » puis « **Récent** ».
- Style factuel, dense, zéro remplissage.
RULES
)"

prompt="$rules

=== RÉCIT ACTUEL ===
$prior

=== NOUVEAUX COMMITS (ancien → récent) ===
$commits
"

# --- generate (bounded; detachment means latency never blocks the committer) ---
new="$(printf '%s' "$prompt" | HIST_ROOT="$root" "$PY_BIN" "$LLM_SCRIPT" 2>>"$LOG")" \
  || { log "llm error"; exit 0; }
new="$(printf '%s\n' "$new" | sed '/^```/d')"        # strip any stray code fences
if [ -z "${new// }" ]; then log "empty output; skip"; exit 0; fi
new="$(printf '%s\n' "$new" | head -n "$MAXLINES")"   # hard safety cap

# --- durable backup (atomic) ---
tmpb="$(mktemp)" && printf '%s\n' "$new" > "$tmpb" && mv "$tmpb" "$BACKUP"

# --- inject into DOC between markers (atomic; rest of the file untouched) ---
NEWBLOCK="$new" BEGINM="$BEGIN" ENDM="$END" HEADM="$HEADING" DOCF="$DOC" python3 - <<'PY'
import os, io, tempfile
doc=os.environ["DOCF"]; b=os.environ["BEGINM"]; e=os.environ["ENDM"]
head=os.environ["HEADM"]; block=os.environ["NEWBLOCK"].rstrip("\n")
marker=f"{b}\n{block}\n{e}\n"
try:
    with open(doc, encoding="utf-8") as f: txt=f.read()
except FileNotFoundError:
    txt=""
if (b in txt) and (e in txt):
    out = txt[:txt.index(b)] + marker + txt[txt.index(e)+len(e):]
else:
    sep = "" if txt=="" else ("\n" if txt.endswith("\n") else "\n\n")
    out = txt + sep + "\n" + head + "\n" + marker
d = os.path.dirname(os.path.abspath(doc)) or "."
fd, tmp = tempfile.mkstemp(dir=d)
with io.open(fd, "w", encoding="utf-8") as f: f.write(out)
os.replace(tmp, doc)
PY

git rev-parse HEAD > "$LASTREF" 2>/dev/null || true
log "histoire updated -> $(git rev-parse --short HEAD)"
exit 0
