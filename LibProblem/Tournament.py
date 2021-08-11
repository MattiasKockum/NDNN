#!/usr/bin/env python3

import random

def play_match(L, P1, P2):
    print("match entre {} et {}".format(P1, P2))
    winner = random.choice([P1, P2])
    print("winner : {}".format(winner))
    L[winner] += 1


def play_tournament_recursive(L, index_min, index_max):
    #print("tournament dans {}".format(L[index_min:index_max+1]))
    l = index_max - index_min + 1
    if l == 1:
        print("{} play seul".format(index_max))
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
        print("winner_left : {} ; winner_right : {}".format(
            winner_left, winner_right))
        play_match(L, winner_right, winner_left)
    print(L)

def play_tournament(L):
    play_tournament_recursive(L, 0, len(L)-1)
