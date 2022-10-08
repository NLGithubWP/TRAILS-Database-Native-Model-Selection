import math
import random

eta = 3


def fun(K_, t1_, t2_, w_):
    N_ = K_ * 100
    B1 = N_ * t1_
    B2 = 0
    current_k = K_

    round_ = 0
    while True:
        if current_k <= 1:
            break
        round_ += 1
        # task > workers
        if current_k >= w_:
            B2 += math.ceil(current_k / w_) * t2_
        else:
            B2 += t2_
        current_k = int(current_k / eta)
    return B1 + B2, B1, B2, round_


def get_K(T_, t1_, t2_, w_):
    x = []
    y = []
    y_ele = 0
    history = []
    for K_ in range(1, int(T_ / t1_)):
        y_ele, B1, B2, round_ = fun(K_, t1_, t2_, w_)
        x.append(K_)
        y.append(y_ele)
        history.append((B1, B2))
        if y_ele > T_:
            print(f"B2 uses {round_} rounds")
            break

    # cannot finish win
    if len(x) < 2:
        print(f"Budget T is too small, it's at least >= {y_ele} with current worker, t1, t2, eta")
        return 0
    else:
        B1 = history[-2][-2]
        B2 = history[-2][-1]
        K = x[-2]
        print(f"When T = {T_}s, B1 = {T_-B2}s and B2 = {B2}s")
        N_ = int(B1/t1_)
        return x[-2], N_


if __name__ == "__main__":
    W = 1
    T = 196 * 60
    t1 = random.randint(3315, 4502) * 0.0001
    t2 = 20
    print(f"t1 = {t1}, t2 = {t2}")
    k, N = get_K(T, t1, t2, W)
    print(f"k = {k}, N = {N}")
