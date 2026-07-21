# Node User Setup — nvm + Node.js + OpenCode

Run these steps as **root** on the new node to install nvm, Node.js, and OpenCode for the `user` account.

## Prerequisites

- `user` account exists
- `curl` installed (`apt install curl -y`)

## Steps

```bash
# 1. Install nvm v0.40.6 for user
su - user -c 'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.6/install.sh | bash'

# 2. Install Node.js v24 (latest 24.x)
su - user -c 'export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && nvm install 24 && nvm alias default 24'

# 3. Install OpenCode globally via npm
su - user -c 'export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && npm install -g opencode-ai'

# 4. Add nvm sourcing to .profile (so it works in non-interactive shells too)
su - user -c 'cat >> ~/.profile << '"'"'EOF'"'"'

# Load nvm (before .bashrc since .bashrc returns early in non-interactive shells)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
EOF'
```

## Verification

```bash
su - user -c 'node --version'
su - user -c 'opencode --version'
```

Expected output:
- `node --version` → `v24.18.0`
- `opencode --version` → `1.18.4`

## Installed Versions (Node A)

| Component | Version |
|---|---|
| nvm | v0.40.6 |
| Node.js | v24.18.0 |
| npm | 11.16.0 |
| OpenCode | 1.18.4 |
