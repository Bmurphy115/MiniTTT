# Train the network here using self-play
import environment as env
import network

# Hyperparameters
game_count = 50000
board_len = 3
games_per_update_opponent = 100
games_per_render = 10000
epsilon = 0.9
epsilon_decay = 0.9997


training_environment = env.Environment(board_len)

model = network.MiniTTT(board_len)
opponent = network.MiniTTT(board_len)

x_player = model
model.symbol = 'X'
o_player = opponent

for game_nr in range(game_count):
    training_environment.reset()
    move_count = 0
    winner = 'N'
    turn = 'X'

    obs = training_environment.board

    while winner == 'N':

        if turn == 'X':
            move = x_player.decide(obs, epsilon)

        else:
            move = o_player.decide(obs, epsilon)

        obs, reward, info = training_environment.step(turn, move)

        winner = info

        if turn == 'X':
            turn = 'O'
        else:
            turn = 'X'

    epsilon *= epsilon_decay

    # Logs:
    print(f'Game {game_nr + 1}, epsilon: {epsilon}')

    if (game_nr + 1) % games_per_render == 0:
        training_environment.render(training_environment.board)

    if (game_nr + 1) % games_per_update_opponent == 0:
        opponent.load_brain(model)

    if winner != 'T':
        model.train(training_environment.board_history[model.symbol], training_environment.move_history[model.symbol], winner == model.symbol, 0.8)


model.save_brain(name='brain50k')





