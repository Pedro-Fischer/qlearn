from connection import connect, get_state_reward
import random
import numpy as np

# Conecta ao jogo
server_socket = connect(2037)
if server_socket == 0:
    print("Erro ao conectar com o jogo. Verifique a porta e se o jogo está rodando.")
    exit()

# Inicializa Q-table ou carrega existente
try:
    utility_matrix = np.loadtxt('resultado.txt')
    print("Q-table carregada de 'resultado.txt'")
except:
    utility_matrix = np.zeros((96, 3))
    print("Q-table nova criada")

np.set_printoptions(precision=6)
actions = ["left", "right", "jump"]

# Hiperparâmetros
alpha = 0.6
gamma = 0.9
epsilon = 0.3
min_epsilon = 0.05
decay_rate = 0.995

successes = 0
episodes = 0
visited_states = set()

max_total_episodes = 5000

while len(visited_states) < 96 and episodes < max_total_episodes:
    print(f"\nEpisódio {episodes + 1}")
    state_info, reward = get_state_reward(server_socket, "jump")
    state = int(state_info[2:], 2)
    total_reward = 0

    while True:
        visited_states.add(state)
        print(f"Estado atual: {state} | Estados cobertos: {len(visited_states)}/96")

        if random.random() < epsilon:
            action_index = random.randint(0, 2)
            chosen_action = actions[action_index]
        else:
            action_index = np.argmax(utility_matrix[state])
            chosen_action = actions[action_index]

        state_info, reward = get_state_reward(server_socket, chosen_action)
        next_state = int(state_info[2:], 2)
        visited_states.add(next_state)

        # Atualização da Q-table (regra de Bellman)
        old_value = utility_matrix[state][action_index]
        next_max = np.max(utility_matrix[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        utility_matrix[state][action_index] = new_value

        total_reward += reward
        state = next_state

        # Atualiza epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        if reward == -100:
            print("Personagem morreu")
            break
        elif reward == 300:
            print("Objetivo atingido!")
            successes += 1
            break

    episodes += 1

    if episodes % 10 == 0:
        print(f"Progresso salvo após {episodes} episódios")
        np.savetxt('resultado.txt', utility_matrix, fmt="%.6f")

print("\nTreinamento encerrado")
print(f"Total de episódios: {episodes}")
print(f"Estados cobertos: {len(visited_states)}/96")
print(f"Taxa de sucesso: {(successes / episodes) * 100:.2f}%")

# Salvar Q-table final
np.savetxt('resultado.txt', utility_matrix, fmt="%.6f")
print("Q-table final salva com sucesso.")
