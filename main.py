import random, math, os
import numpy as np
from plot import plot_fitness, plot_contours, make_gif, plot_diversity

# parâmetros
POP_SIZE, NGEN = 100, 500
BITS_X, BITS_Y = 25, 25
X_MIN, X_MAX, Y_MIN, Y_MAX = -100, 100, -100, 100
CXPB, MUTPB = 0.8, 0.01  # MUTPB

# função F6
def f6_scalar(x, y):
    num = math.sin(math.sqrt(x*x + y*y))**2 - 0.5
    den = (1.0 + 0.001*(x*x + y*y))**2
    return 0.5 - num/den

def f6_numpy(X, Y):
    R2 = X**2 + Y**2
    return 0.5 - ((np.sin(np.sqrt(R2))**2) - 0.5) / ((1.0 + 0.001*R2)**2)

# utilitários do AG
def decode_bits(bits, min_v, max_v):
    intval = int("".join(map(str, bits)), 2)
    return min_v + (max_v-min_v) * intval / (2**len(bits)-1)

def decode_ind(ind):
    x = decode_bits(ind[:BITS_X], X_MIN, X_MAX)
    y = decode_bits(ind[BITS_X:], Y_MIN, Y_MAX)
    return x, y

def fitness(ind):
    x, y = decode_ind(ind)
    return f6_scalar(x, y)

def random_ind():
    return [random.randint(0,1) for _ in range(BITS_X+BITS_Y)]

def roulette(pop):
    fits = [fitness(ind) for ind in pop]
    total = sum(fits)
    cum = np.cumsum(fits)
    chosen = []
    for _ in range(len(pop)):
        r = random.uniform(0,total)
        idx = np.searchsorted(cum,r)
        chosen.append(pop[idx][:])
    return chosen

def crossover(p1,p2):
    if random.random()<CXPB:
        pt = random.randint(1,len(p1)-1)
        return p1[:pt]+p2[pt:], p2[:pt]+p1[pt:]
    return p1[:],p2[:]

def mutate(ind, mutpb):
    for i in range(len(ind)):
        if random.random()<mutpb:
            ind[i]=1-ind[i]
    return ind

# diversidade
def population_diversity(pop):
    """ Calcula a diversidade média (distância euclidiana média entre indivíduos). """
    pop_xy = np.array([decode_ind(ind) for ind in pop])
    N = len(pop_xy)
    if N < 2:
        return 0.0
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pop_xy[i] - pop_xy[j])
            dists.append(d)
    return np.mean(dists)

def main():
    pop = [random_ind() for _ in range(POP_SIZE)]
    best_fits, avg_fits, worst_fits = [], [], []
    diversities, mut_rates = [], []
    all_pop_xy = []

    for g in range(NGEN+1):
        fits = [fitness(ind) for ind in pop]
        best_fits.append(max(fits))
        avg_fits.append(np.mean(fits))
        worst_fits.append(min(fits))
        all_pop_xy.append(np.array([decode_ind(ind) for ind in pop]))

        # diversidade da população
        div = population_diversity(pop)
        diversities.append(div)

        # mutação adaptativa
        mutpb = 0.001 + (0.05 / (1.0 + div))
        mut_rates.append(mutpb)

        if g < NGEN:
            offspring = roulette(pop)
            newpop=[]
            for p1,p2 in zip(offspring[::2],offspring[1::2]):
                c1,c2=crossover(p1,p2)
                newpop.append(mutate(c1, mutpb))
                newpop.append(mutate(c2, mutpb))
            pop=newpop[:POP_SIZE]

    # cria results para guardar os gráficos
    os.makedirs("results", exist_ok=True)

    # === chama funções de plotagem ===
    plot_fitness(best_fits, avg_fits, worst_fits, outpath="results/fitness.png")
    plot_contours(f6_numpy, X_MIN, X_MAX, Y_MIN, Y_MAX, all_pop_xy[-1], outpath="results/contours.png")
    make_gif(f6_numpy, X_MIN, X_MAX, Y_MIN, Y_MAX, all_pop_xy, outpath="results/evolution.gif", frame_every=5)
    plot_diversity(diversities, mut_rates, avg_fits, outpath="results/diversity.png")

    #print("Resultados salvos na pasta 'results/'")


if __name__=="__main__":
    main()
