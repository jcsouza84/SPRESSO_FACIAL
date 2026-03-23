#!/usr/bin/env bash
# Verifica se o Raspberry Pi está acessível na rede
# Uso: ./check_pi.sh
# Rode este script do SEU COMPUTADOR, não do Pi

PI_IP="192.168.84.131"
PI_USER="spresso"
SSH_PORT=22

echo "========================================"
echo "  SPRESSO PI — Verificação de acesso"
echo "========================================"
echo ""

# 1. Ping
echo -n "1. Ping ($PI_IP)... "
if ping -c 2 -W 2 "$PI_IP" &>/dev/null; then
    echo "✅ Online"
else
    echo "❌ Sem resposta — Pi pode estar desligado ou fora da rede"
    exit 1
fi

# 2. Porta SSH
echo -n "2. Porta SSH ($SSH_PORT)... "
if nc -z -w 3 "$PI_IP" "$SSH_PORT" &>/dev/null; then
    echo "✅ Aberta"
else
    echo "❌ Fechada — serviço SSH pode estar parado"
    exit 1
fi

# 3. API SPRESSO
echo -n "3. API SPRESSO (porta 8000)... "
if curl -s --max-time 3 "http://$PI_IP:8000/health" &>/dev/null; then
    HEALTH=$(curl -s --max-time 3 "http://$PI_IP:8000/health")
    echo "✅ Online — $HEALTH"
else
    echo "⚠️  Sem resposta (API pode não estar rodando ainda)"
fi

echo ""
echo "Para conectar via SSH:"
echo "  ssh $PI_USER@$PI_IP"
echo ""
