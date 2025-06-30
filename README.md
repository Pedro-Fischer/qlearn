
# Q-Learning – Projeto de Aprendizado por Reforço com Amongois

Este projeto implementa o algoritmo **Q-Learning** para treinar um agente capaz de controlar o personagem **Amongois** em um jogo. O objetivo é que o agente aprenda, por tentativa e erro, o melhor caminho até o **bloco preto**, evitando cair e recebendo recompensas de acordo com suas ações.

---

## Como Executar o Projeto

### 1. Inicie o Jogo

- Execute o arquivo `.exe` do jogo Amongois.
- Verifique a **porta TCP** exibida na janela do jogo (ex: `2037`).

> Essa porta deve ser usada no script de treinamento para estabelecer a conexão.

### 2. Execute o Script de Treinamento

- No terminal, com o ambiente Python ativado e o jogo aberto, execute:

```bash
python client.py
```

Esse script:

- Conecta ao jogo via socket.
- Treina o agente a partir de episódios sucessivos.
- Armazena a Q-table aprendida no arquivo `resultado.txt`.
- Encerra o treinamento automaticamente quando todos os **96 estados** forem visitados ou quando atingir o número máximo de episódios.

> A Q-table é salva automaticamente a cada 10 episódios e ao final do treinamento.

---

## Como Funciona o Q-Learning

O Q-Learning é um algoritmo de **aprendizado por reforço**. O agente interage com o ambiente, executa ações e aprende com base nas recompensas recebidas.

### Fórmula de atualização (Equação de Bellman):

```
new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
```

- `alpha`: taxa de aprendizado
- `gamma`: fator de desconto
- `reward`: recompensa recebida

A Q-table guarda os valores esperados de cada ação em cada estado, e é usada para decidir as ações futuras.

---

## Estratégia de Aprendizado

- **Estados**: representados por números inteiros de 0 a 95 (convertidos do binário recebido do jogo).
- **Ações disponíveis:** `"left"`, `"right"`, `"jump"`
- **Recompensas:**
  - `-1 a -14`: penalidades por movimentações comuns
  - `300`: objetivo alcançado
  - `-100`: morte (queda)

### Política Epsilon-Greedy

- Ação aleatória com probabilidade `epsilon` (exploração).
- Melhor ação segundo a Q-table com probabilidade `1 - epsilon` (exploração).
- `epsilon` decai a cada passo até o mínimo de `0.05`, para garantir aprendizado estável e ampla cobertura de estados.

---

## 🗂️ Arquivos do Projeto

| Arquivo                      | Descrição |
|------------------------------|-----------|
| `client.py`                  | Script de treinamento e preenchimento da Q-table |
| `connection.py`              | Interface de comunicação com o jogo via socket TCP |
| `resultado.txt`              | Q-table contendo os valores aprendidos para cada estado e ação |

---

## ⚙️ Hiperparâmetros

| Parâmetro         | Valor  |
|-------------------|--------|
| `alpha`           | 0.6    |
| `gamma`           | 0.9    |
| `epsilon` inicial | 0.3    |
| `min_epsilon`     | 0.05   |
| `decay_rate`      | 0.995  |
| `episódios máx.`  | 5000   |

---

## Resultados

- A Q-table aprendida cobre os 96 estados possíveis do ambiente.
- O agente é capaz de alcançar o objetivo após treinamento suficiente.

---

## Entregáveis

- `client.py`
- `connection.py`
- `resultado.txt` (Q-table final)
- vídeo demonstrando o projeto.

---

## Grupo

Pedro de Andrade Lima Fischer de Lyra
Ivo Luiz Soares Neto
Walter Crasto Monteiro
Daniel Dias Fernandes
