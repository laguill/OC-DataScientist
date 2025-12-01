#!/bin/bash
set -e

echo "=========================================="
echo "Bootstrap EMR avec uv (utilisateur hadoop, chemin custom)"
echo "Date: $(date)"
echo "=========================================="

HADOOP_HOME="/home/hadoop"
UV_DIR="$HADOOP_HOME/.local/uv"
PROJECT_DIR="$HADOOP_HOME/project"
MARIMO_DIR="$PROJECT_DIR/marimo-notebooks"
S3_BUCKET="s3://p9-fruits-data/project/marimo-notebooks"

# -------------------------------------------------------------------
# 1. Installation uv pour l'utilisateur hadoop
# -------------------------------------------------------------------
echo "[+] Installation uv dans $UV_DIR"
sudo -u hadoop HOME=$HADOOP_HOME bash -lc "
  mkdir -p $UV_DIR
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR='$UV_DIR' sh
"

# Ajouter uv au PATH automatiquement
echo "source $UV_DIR/env" >> $HADOOP_HOME/.bashrc

# Test de l'installation
sudo -u hadoop HOME=$HADOOP_HOME bash -lc "
  source $UV_DIR/env
  uv --version
"

# -------------------------------------------------------------------
# 2. Creation du projet uv avec Python 3.12
# -------------------------------------------------------------------
echo "[+] Installation de Python 3.12..."
sudo -u hadoop HOME=$HADOOP_HOME bash -lc "
  source $UV_DIR/env
  uv python install 3.12
"

echo "[+] Creation du projet uv avec Python 3.12..."
sudo -u hadoop HOME=$HADOOP_HOME bash -lc "
  mkdir -p $PROJECT_DIR
  cd $PROJECT_DIR
  source $UV_DIR/env
  uv init --app --python 3.12
"

# -------------------------------------------------------------------
# 3. Installation des dependances Python
# -------------------------------------------------------------------
echo "[+] Installation des dependances uv..."
sudo -u hadoop HOME=$HADOOP_HOME bash -lc "
  cd $PROJECT_DIR
  source $UV_DIR/env
  uv add pillow pandas pyarrow boto3 s3fs fsspec matplotlib 'marimo[lsp]' ruff
"

# -------------------------------------------------------------------
# 4. Synchronisation Marimo notebooks depuis S3
# -------------------------------------------------------------------
echo "[+] Synchronisation Marimo notebooks..."
sudo -u hadoop HOME=$HADOOP_HOME bash -lc "
  mkdir -p $MARIMO_DIR
  aws s3 sync $S3_BUCKET $MARIMO_DIR/ || true
"

# -------------------------------------------------------------------
# 5. Ajouter aliases et activation automatique uv dans .bashrc
# -------------------------------------------------------------------
cat >> $HADOOP_HOME/.bashrc << 'EOF'

# uv dans le PATH
export PATH="$HOME/.local/uv/bin:$PATH"

# activer automatiquement l'environnement uv du projet
if [ -d "$HOME/project/.venv" ]; then
    source "$HOME/project/.venv/bin/activate"
fi

# aliases Marimo
alias save='aws s3 sync ~/project/marimo-notebooks/ s3://p9-fruits-data/project/marimo-notebooks/'
alias load='aws s3 sync s3://p9-fruits-data/project/marimo-notebooks/ ~/project/marimo-notebooks/'
EOF

# -------------------------------------------------------------------
# 6. Cron autosave pour Marimo (correction: sudo crontab -u)
# -------------------------------------------------------------------
echo "[+] Configuration du cron autosave..."
echo "*/15 * * * * aws s3 sync $MARIMO_DIR $S3_BUCKET --quiet" | sudo crontab -u hadoop -

# Verifier le cron
echo "[+] Verification du cron:"
sudo crontab -u hadoop -l

echo "=========================================="
echo "Bootstrap uv termine avec succes"
echo "=========================================="

exit 0
