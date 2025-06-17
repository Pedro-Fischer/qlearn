import random
import numpy as np
import time
import json
from connection import connect, get_state_reward

# ==========================
# 🔧 Hiperparâmetros Otimizados para Velocidade
# ==========================
ALPHA = 0.1           # Taxa de aprendizado mais alta (aprendizado mais rápido)
GAMMA = 0.9           # Fator de desconto
EPSILON_START = 0.5   # Taxa de exploração inicial (menor)
EPSILON_END = 0.05    # Taxa de exploração final
EPSILON_DECAY = 0.99  # Decaimento mais rápido do epsilon
NUM_EPISODES = 1000    # Número reduzido de episódios
MAX_STEPS = 100       # Máximo de passos por episódio (reduzido)
PORTA = 2037          # Porta do jogo

# ==========================
# 🎯 Configurações do ambiente
# ==========================
NUM_STATES = 96       # 24 plataformas × 4 direções
NUM_ACTIONS = 3       # left, right, jump
ACTIONS = ["left", "right", "jump"]

# Mapeamento de ações para índices
ACTION_TO_INDEX = {action: i for i, action in enumerate(ACTIONS)}

# ==========================
# 📊 Classe para estatísticas
# ==========================
class TrainingStats:
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0
        self.epsilon_history = []
    
    def add_episode(self, total_reward, steps, success, epsilon):
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.epsilon_history.append(epsilon)
        if success:
            self.success_count += 1
    
    def print_stats(self, episode):
        if episode % 25 == 0 and episode > 0:  # Stats mais frequentes
            recent_rewards = self.episode_rewards[-25:] if len(self.episode_rewards) >= 25 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(self.episode_steps[-25:]) if len(self.episode_steps) >= 25 else np.mean(self.episode_steps)
            success_rate = (self.success_count / episode) * 100
            print(f"📊 Ep {episode}: Avg Reward: {avg_reward:.1f} | Steps: {avg_steps:.0f} | Success: {success_rate:.1f}% | ε: {self.epsilon_history[-1]:.2f}")

# ==========================
# 🧠 Classe Q-Learning Agent
# ==========================
class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=ALPHA, gamma=GAMMA):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = EPSILON_START
        
    def state_to_index(self, state_binary):
        """
        Converte estado binário para índice (0-95).
        Valida se o estado está no range correto.
        """
        try:
            index = int(state_binary, 2)
            if 0 <= index < self.num_states:
                return index
            else:
                print(f"⚠️ Estado fora do range: {index}")
                return 0  # Estado padrão
        except ValueError:
            print(f"⚠️ Estado inválido: {state_binary}")
            return 0
    
    def choose_action(self, state_index):
        """
        Escolhe ação usando estratégia ε-greedy.
        """
        if random.random() < self.epsilon:
            # Exploração: ação aleatória
            return random.choice(ACTIONS)
        else:
            # Exploração: melhor ação conhecida
            best_action_index = np.argmax(self.q_table[state_index])
            return ACTIONS[best_action_index]
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Atualiza Q-table usando a fórmula do Q-Learning.
        """
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        action_idx = ACTION_TO_INDEX[action]
        
        # Q-Learning: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state_idx, action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_idx, action_idx] = new_q
    
    def decay_epsilon(self):
        """
        Reduz epsilon gradualmente (menos exploração, mais exploração).
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save_q_table(self, filename="resultado.txt"):
        """
        Salva Q-table no formato especificado:
        - Apenas dados numéricos
        - Ordenados por estado (0-95)
        - Colunas: [left, right, jump]
        """
        try:
            with open(filename, "w") as file:
                for state_idx in range(self.num_states):
                    row = self.q_table[state_idx]
                    # Formatar com 6 casas decimais para precisão
                    line = " ".join([f"{value:.6f}" for value in row])
                    file.write(line + "\n")
            print(f"✅ Q-table salva em '{filename}'")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar Q-table: {e}")
            return False
    
    def load_q_table(self, filename="resultado.txt"):
        """
        Carrega Q-table de arquivo (útil para continuar treinamento).
        """
        try:
            self.q_table = np.loadtxt(filename)
            print(f"✅ Q-table carregada de '{filename}'")
            return True
        except Exception as e:
            print(f"⚠️ Não foi possível carregar Q-table: {e}")
            return False

# ==========================
# 🎮 Função principal de treinamento
# ==========================
def train_agent():
    """
    Treina o agente Q-Learning no jogo Amongois.
    """
    print("🚀 Iniciando treinamento RÁPIDO do Q-Learning Agent")
    print(f"📋 Configurações otimizadas para velocidade:")
    print(f"   Episódios: {NUM_EPISODES} (reduzido)")
    print(f"   Max steps: {MAX_STEPS} (reduzido)")
    print(f"   Alpha: {ALPHA} (mais alto)")
    print(f"   Epsilon: {EPSILON_START} → {EPSILON_END} (decaimento rápido)")
    print("=" * 50)
    
    # Inicializar agente e estatísticas
    agent = QLearningAgent(NUM_STATES, NUM_ACTIONS)
    stats = TrainingStats()
    
    # Tentar carregar Q-table existente
    agent.load_q_table()
    
    for episode in range(1, NUM_EPISODES + 1):
        # Conectar ao jogo
        socket_conn = connect(PORTA)
        if socket_conn == 0:
            print("❌ Falha na conexão. Verifique se o jogo está rodando.")
            break
        
        # Obter estado inicial
        try:
            initial_state, _ = get_state_reward(socket_conn, "jump")
            current_state = initial_state
        except Exception as e:
            print(f"❌ Erro ao obter estado inicial: {e}")
            socket_conn.close()
            continue
        
        # Variáveis do episódio
        total_reward = 0
        steps = 0
        episode_success = False
        
        # Loop do episódio
        for step in range(MAX_STEPS):
            # Escolher e executar ação
            action = agent.choose_action(agent.state_to_index(current_state))
            
            try:
                next_state, reward = get_state_reward(socket_conn, action)
            except Exception as e:
                print(f"❌ Erro na comunicação: {e}")
                break
            
            # Atualizar Q-table
            agent.update_q_value(current_state, action, reward, next_state)
            
            # Atualizar métricas
            total_reward += reward
            steps += 1
            
            # Verificar condições de parada
            if reward == -1:  # Chegou ao objetivo
                episode_success = True
                if episode % 25 == 0:  # Só mostra sucesso a cada 25 episódios
                    print(f"🎉 Ep {episode}: SUCESSO em {steps} passos!")
                break
            elif reward <= -12:  # Penalidade muito alta (mais restritiva)
                break
            
            current_state = next_state
        
        # Fechar conexão
        socket_conn.close()
        
        # Decair epsilon
        agent.decay_epsilon()
        
        # Registrar estatísticas
        stats.add_episode(total_reward, steps, episode_success, agent.epsilon)
        stats.print_stats(episode)
        
        # Salvar Q-table periodicamente (menos frequente)
        if episode % 50 == 0:
            agent.save_q_table(f"checkpoint_ep{episode}.txt")
    
    # Salvar Q-table final
    agent.save_q_table()
    
    # Estatísticas finais
    print("\n🏁 TREINAMENTO CONCLUÍDO!")
    print(f"📊 Estatísticas finais:")
    print(f"   Total de sucessos: {stats.success_count}/{NUM_EPISODES}")
    print(f"   Taxa de sucesso: {(stats.success_count/NUM_EPISODES)*100:.1f}%")
    print(f"   Recompensa média: {np.mean(stats.episode_rewards):.2f}")
    print(f"   Epsilon final: {agent.epsilon:.3f}")
    
    return agent

# ==========================
# 🎯 Função para testar agente treinado
# ==========================
def test_agent(num_tests=5):  # Menos testes
    """
    Testa o agente já treinado (sem exploração) - versão rápida.
    """
    print("🧪 Testando agente treinado (teste rápido)...")
    
    agent = QLearningAgent(NUM_STATES, NUM_ACTIONS)
    agent.epsilon = 0  # Sem exploração
    
    if not agent.load_q_table():
        print("❌ Não foi possível carregar Q-table para teste.")
        return
    
    successes = 0
    
    for test in range(1, num_tests + 1):
        socket_conn = connect(PORTA)
        if socket_conn == 0:
            continue
        
        try:
            current_state, _ = get_state_reward(socket_conn, "jump")
            steps = 0
            
            for step in range(MAX_STEPS):
                action = agent.choose_action(agent.state_to_index(current_state))
                next_state, reward = get_state_reward(socket_conn, action)
                
                steps += 1
                if reward == -1:
                    successes += 1
                    print(f"✅ Teste {test}: Sucesso em {steps} passos")
                    break
                elif reward <= -12:
                    print(f"❌ Teste {test}: Falha em {steps} passos")
                    break
                
                current_state = next_state
                # Removida a pausa para teste mais rápido
        
        except Exception as e:
            print(f"❌ Erro no teste {test}: {e}")
        
        socket_conn.close()
    
    print(f"🎯 Resultado dos testes: {successes}/{num_tests} sucessos ({(successes/num_tests)*100:.1f}%)")

# ==========================
# 🚀 Execução principal
# ==========================
if __name__ == "__main__":
    try:
        # Treinar agente
        trained_agent = train_agent()
        
        # Perguntar se quer testar
        test_choice = input("\n🤔 Deseja testar o agente treinado? (s/n): ")
        if test_choice.lower() == 's':
            test_agent()
    
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
