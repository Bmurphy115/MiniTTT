import game_mechanics as gm

t = gm.create_board(3)

gm.select_space('X', t, (0, 0))
# gm.select_space('O', t, (0, 1))
# gm.select_space('X', t, (1, 0))
# gm.select_space('O', t, (1, 1))
# gm.select_space('X', t, (2, 1))
# gm.select_space('O', t, (1, 2))
# gm.kelect_space('X', t, (0, 2))
# gm.select_space('X', t, (2, 2))
# gm.select_space('O', t, (2, 0))

print(gm.game_is_over(t))










