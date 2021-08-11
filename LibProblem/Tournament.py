#!/usr/bin/env python3

import random

def play_match(L, P1, P2):
    winner = random.choice([P1, P2])
    L[winner] += 1


def play_tournament_recursive(L, index_min, index_max):
    l = index_max - index_min + 1
    if l == 1:
        L[index_min] += 1
    elif l == 2:
        play_match(L, index_min, index_max)
    else:
        play_tournament_recursive(L, index_min, index_min+int(l/2)-1)
        play_tournament_recursive(L, index_min+int(l/2), index_max)
        Lr = L[index_min : index_min+int(l/2)]
        Ll = L[index_min+int(l/2) : index_max+1]
        winner_left = Lr.index(max(Lr)) + index_min
        winner_right = Ll.index(max(Ll)) + index_min + int(l/2)
        play_match(L, winner_right, winner_left)

def play_tournament(L):
    play_tournament_recursive(L, 0, len(L)-1)
