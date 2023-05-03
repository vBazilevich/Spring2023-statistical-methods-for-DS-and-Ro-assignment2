import random
import math as m
from typing import Dict, Tuple
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

EARTH_RADIUS = 6371  # km

def extract_cities() -> Dict[str, list[float]]:
    """
        Extracts city name and coordinates for 30 most populated cities in Russia
    """
    cities = pd.read_csv('city.csv').sort_values('population', ascending=False)

    # Handle federal cities
    cities.city.fillna(cities.region, inplace=True)
    cities = cities[['city', 'geo_lat', 'geo_lon']].head(30)
    cities.set_index('city', drop=True, inplace=True)
    return cities.T.to_dict('list')

def haversine_distance(first_city : Tuple[float, float], second_city : Tuple[float, float]) -> float:
    """
        Takes coordinates of two cities in format (lat, lon) and returns great-circle distance between them
    """
    phi1, lambda1 = first_city
    phi1, lambda1 = m.radians(phi1), m.radians(lambda1)
    phi2, lambda2 = second_city
    phi2, lambda2 = m.radians(phi2), m.radians(lambda2)

    return 2 * EARTH_RADIUS * m.asin(m.sqrt(
            m.sin((phi2 - phi1) / 2)**2 + m.cos(phi1) * m.cos(phi2) * m.sin((lambda2 - lambda1)/2)**2
        ))

def compute_distance_matrix(cities : Dict[str, list[float]]) -> Dict[str, Dict[str, float]]:
    """
        Computes great-circle distance between all pairs of cities.
        Cities is expected to be a dictionary having city name as a key and (lat, lon) pair as a value
    """
    return {u: {v:haversine_distance(cities[u], cities[v]) for v in cities.keys()} for u in cities.keys()}

def compute_path_length(path: list[str], distances: Dict[str, Dict[str, float]]):
    path_len = distances[path[0]][path[-1]]
    for i in range(len(path) - 1):
        path_len += distances[path[i]][path[i+1]]
    return path_len

def generate_candidate(path: list[str]):
    """
        Swap two random cities in a given path to obtain a new candidate
    """
    N = len(path)
    i, j = random.sample(range(len(path)), 2)
    path[i], path[j] = path[j], path[i]
    return path

class SimulatedAnnealingSolver:
    def __init__(self, temperature_decay, distances, update_rate=1):
        assert 0 < temperature_decay and temperature_decay < 1, "Temperature decay rate must be in (0, 1) range"

        self.__temperature_decay = temperature_decay
        self.__distances = distances
        self.__update_rate = update_rate

    def solve(self, initial_state, cities):
        timestamp = 0
        state = initial_state
        temperature = compute_path_length(state, self.__distances)
        shortest_path = initial_state
        shortest_paths = [shortest_path[:]]

        while temperature > 1:
            candidate = generate_candidate(shortest_path[:])

            # Sometimes exponentiation result might be outside of the double range and cause the overflow
            try:
                acceptance_ratio = m.exp((compute_path_length(shortest_path, distances) - compute_path_length(candidate, distances)) / temperature)
                if random.random() <= acceptance_ratio:
                    shortest_path = candidate[:]
            except:
                pass
            if timestamp % self.__update_rate == 0:
                temperature *= self.__temperature_decay
            timestamp += 1
            shortest_paths.append(shortest_path[:])
        # Plot decay and save to the file
        plt.plot([compute_path_length(p, distances) for p in shortest_paths])
        plt.title('Shortest path length')
        plt.ylabel('Path length, km')
        plt.xlabel('t')
        plt.savefig(f'{self}.png')
        return shortest_paths

    def __str__(self):
        return f"sa_td{self.__temperature_decay}_ur{self.__update_rate}"

if __name__ == "__main__":
    cities = extract_cities()
    distances = compute_distance_matrix(cities)

    solver = SimulatedAnnealingSolver(0.99, distances, 5)
    paths = solver.solve(list(cities.keys()), cities)

    # Draw animation as per this tutorial http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    fig = plt.figure()
    ax = plt.axes(xlim=(25, 140), ylim=(40, 63))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        path = paths[i]
        coords = [cities[c] for c in path]
        x = [c[1] for c in coords]
        y = [c[0] for c in coords]
        line.set_data(x, y)
        ax.set_title(f'Current path length: {compute_path_length(path, distances):.3f}')
        return line,

    plt.title("Simulated Annealing TSP")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(paths), interval=20, blit=True)
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
