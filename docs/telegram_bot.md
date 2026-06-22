# Telegram Bot — Configuração SPRESSO FACIAL

## Por que Telegram em vez de WhatsApp?

| Critério | Telegram Bot (BotFather) | Evolution API / WhatsApp |
|---|---|---|
| API oficial | Sim | Não (automação de WhatsApp Web) |
| Risco de bloqueio | Praticamente zero | Alto (Meta detecta bots Baileys) |
| Sessão frágil (QR Code) | Não — token estático | Sim — requer reconexão periódica |
| HTTPS nativo | Sim (`api.telegram.org`) | Não (VPS sem TLS) |
| Infraestrutura própria | Não — CDN global do Telegram | Sim — VPS externo (ponto de falha) |
| Envio de foto | `sendPhoto` (bytes diretos) | `sendMedia` (base64 pesado) |
| Grupos | Sim | Não |
| Custo | Gratuito | VPS Hostinger (~R$ 40–80/mês) |

---

## Criando o bot via @BotFather

1. Abra o Telegram e pesquise por **@BotFather**
2. Inicie a conversa com `/start`
3. Digite `/newbot`
4. Informe um nome para o bot (ex: `SPRESSO Alertas`)
5. Informe um username para o bot — deve terminar em `bot` (ex: `spresso_alertas_bot`)
6. O BotFather retornará uma mensagem com o **token** no formato:
   ```
   123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
7. Copie o token — ele será usado na configuração do SPRESSO

---

## Descobrindo o Chat ID

O Telegram usa `chat_id` como destino, não número de telefone.

### Para receber alertas individualmente:
1. Abra o Telegram e pesquise pelo username do bot que criou
2. Clique em **Start** ou envie qualquer mensagem (ex: `oi`)
3. No SPRESSO, acesse **Configurações → Integrações → Telegram**
4. Cole o token do bot no campo **Token do Bot**
5. Clique em **Descobrir** — o sistema chama `getUpdates` e encontra automaticamente o `chat_id`
6. O campo **Chat ID** será preenchido automaticamente

### Para receber alertas em um grupo:
1. Crie um grupo no Telegram com os operadores
2. Adicione o bot ao grupo (pesquise pelo username)
3. Envie qualquer mensagem no grupo
4. Clique em **Descobrir** — o `chat_id` do grupo aparecerá com o nome do grupo

### Múltiplos destinatários:
O campo **Chat ID** aceita múltiplos valores separados por vírgula:
```
7363137004,987654321,-100123456789
```
(IDs de grupos têm prefixo `-100`)

---

## Configuração no SPRESSO

1. Acesse `http://<ip-pi>:8000`
2. Vá em **Configurações → Integrações → Telegram**
3. Preencha:
   - **Token do Bot**: cole o token do BotFather
   - **Chat ID**: clique em **Descobrir** após enviar mensagem ao bot
   - Ative o toggle **Notificações ativas**
4. Clique em **Salvar**
5. Clique em **Testar conexão** — deve mostrar "Bot conectado: @username"
6. Clique em **Enviar mensagem** para confirmar o recebimento

---

## Endpoints da API Telegram usados pelo SPRESSO

| Método | Endpoint | Uso |
|---|---|---|
| `GET` | `/bot{TOKEN}/getMe` | Verificar token e nome do bot |
| `GET` | `/bot{TOKEN}/getUpdates` | Descobrir chat_ids dos remetentes |
| `POST` | `/bot{TOKEN}/sendPhoto` | Enviar alerta com foto do rosto |
| `POST` | `/bot{TOKEN}/sendMessage` | Enviar alerta de texto (sem foto) |

---

## Formato do alerta enviado

```
⚠️ ALERTA BLACKLIST
*João da Silva* identificado(a)
Confiança: 94%
Hora: 22/06 00:30:15
```

Quando a foto do rosto estiver disponível, ela é enviada como imagem com a mensagem na legenda.

---

## Fluxo de notificação

```
detection_worker (Pi)
  └─ _run_pipeline() a cada 1s
       └─ alert_service.process_recognition()
            └─ [somente categoria blacklist]
                 └─ cooldown ok? (padrão: 300s por pessoa)
                      └─ asyncio.gather(
                           _send_telegram(alert),   ← primário
                           _send_whatsapp(alert),   ← secundário / fallback
                         )
```

As duas notificações são disparadas em paralelo — se o Telegram falhar, o WhatsApp ainda tenta (e vice-versa).

---

## Observações operacionais

- **Cooldown:** 300 segundos por pessoa (editável em Configurações)
- **`telegram_sent`:** campo no banco registra se o envio teve sucesso para auditoria
- **Grupos privados:** o bot precisa ser adicionado ao grupo e ter permissão para enviar mensagens
- **Token estático:** ao contrário do WhatsApp, o token do Telegram não expira — não é necessário reconectar
- **Revogação de token:** se necessário, use `/revoke` no @BotFather para gerar um novo token
