import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# Função para gráfico fitness
def plot_fitness(best_fits, avg_fits, worst_fits, outpath="fitness.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(best_fits, label="Melhor")
    plt.plot(avg_fits, label="Média")
    plt.plot(worst_fits, label="Pior")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Evolução do Fitness")
    plt.legend()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()


# Função para curvas de nível + população final
def plot_contours(f6_numpy, X_MIN, X_MAX, Y_MIN, Y_MAX, final_pop, outpath="contours.png"):
    grid_n = 300
    xs = np.linspace(X_MIN, X_MAX, grid_n)
    ys = np.linspace(Y_MIN, Y_MAX, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = f6_numpy(XX, YY)

    plt.figure(figsize=(10, 8))

    # pontos da população
    plt.scatter(final_pop[:, 0], final_pop[:, 1], s=50, c="red", alpha=0.7,
                edgecolors='white', linewidth=0.5, label="População")

    # curvas de nível
    cs = plt.contour(XX, YY, ZZ, levels=25, cmap='viridis', alpha=0.5, linewidths=1.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    # Melhor indivíduo
    from main import f6_scalar
    fitness_values = [f6_scalar(x, y) for x, y in final_pop]
    best_idx = np.argmax(fitness_values)
    best_x, best_y = final_pop[best_idx]
    best_fitness = fitness_values[best_idx]

    plt.scatter(best_x, best_y, s=400, c="gold", marker="*",
                edgecolors="black", linewidth=3, zorder=100,
                label=f"Melhor: F={best_fitness:.4f}")

    circle = plt.Circle((best_x, best_y), 8, color='yellow', fill=False,
                        linestyle='--', linewidth=2, alpha=0.8, zorder=99)
    plt.gca().add_patch(circle)

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title(f"Curvas de Nível - Melhor Fitness: {best_fitness:.4f}", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.2)

    plt.xlim(X_MIN*1.1, X_MAX*1.1)
    plt.ylim(Y_MIN*1.1, Y_MAX*1.1)

    plt.colorbar(cs, label='Fitness', shrink=0.8)
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f" Melhor indivíduo: x={best_x:.4f}, y={best_y:.4f}, fitness={best_fitness:.6f}")


# Função para gerar GIF da evolução
def make_gif(f6_numpy, X_MIN, X_MAX, Y_MIN, Y_MAX, all_pop_xy_per_gen,
             outpath="evolution.gif", frame_every=5):
    grid_n = 200
    xs = np.linspace(X_MIN, X_MAX, grid_n)
    ys = np.linspace(Y_MIN, Y_MAX, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = f6_numpy(XX, YY)

    frames = []
    tmp_dir = "frames_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for g, pop_xy in enumerate(all_pop_xy_per_gen):
        if g % frame_every != 0 and g not in (0, len(all_pop_xy_per_gen)-1):
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(pop_xy[:, 0], pop_xy[:, 1], s=40, c="red", alpha=0.7,
                   edgecolors='white', linewidth=0.5, label="População")

        cs = ax.contour(XX, YY, ZZ, levels=15, cmap='viridis', alpha=0.5, linewidths=1.0)

        from main import f6_scalar
        fitness_values = [f6_scalar(x, y) for x, y in pop_xy]
        best_idx = np.argmax(fitness_values)
        best_x, best_y = pop_xy[best_idx]
        best_fitness = fitness_values[best_idx]

        ax.scatter(best_x, best_y, s=200, c="gold", marker="*",
                   edgecolors="black", linewidth=2, zorder=100,
                   label=f"Melhor: F={best_fitness:.3f}")

        ax.plot([0], [0], marker="X", markersize=8, color="blue", linestyle="",
                label="Ótimo (0,0)")

        ax.set_xlim(X_MIN*1.1, X_MAX*1.1)
        ax.set_ylim(Y_MIN*1.1, Y_MAX*1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Geração {g} - Melhor F: {best_fitness:.4f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        frame_path = os.path.join(tmp_dir, f"frame_{g:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

        print(f"Gerado frame {g}: F_max = {best_fitness:.4f}")

    imageio.mimsave(outpath, frames, duration=0.3, loop=0)
    #print(f"GIF salvo em {outpath} com {len(frames)} frames")


# plotando diversidade e Mutação
def plot_diversity(diversities, mut_rates, avg_fits, outpath="diversity.png"):
    gens = range(len(diversities))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # eixo da esquerda - diversidade
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Diversidade", color="tab:blue")
    ax1.plot(gens, diversities, label="Diversidade", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # eixo da direita - taxa de mutação
    ax2 = ax1.twinx()
    ax2.set_ylabel("Taxa de Mutação", color="tab:red")
    ax2.plot(gens, mut_rates, label="Mutação", color="tab:red", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # média do fitness como linha pontilhada preta
    ax1.plot(gens, avg_fits, color="black", linestyle=":", label="Fitness Médio")

    fig.tight_layout()
    fig.suptitle("Diversidade, Mutação e Fitness Médio", fontsize=14, fontweight="bold", y=1.02)

    # Legenda combinada
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    ax1.legend(lines, labels, loc="upper right")

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em {outpath}")
