# potential.py - potentialization algorithm code
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from misc import idx_from_actions, actions_from_idx


class Potential:
    def __init__(self, m):
        self.utility_matrix = np.array(m)
        if (len(self.utility_matrix.shape) - 1) != self.utility_matrix.shape[-1]:
            raise ValueError("Number of utilities does not match number of players")
        self.num_players = self.utility_matrix.shape[-1]
        self.num_actions = self.utility_matrix.shape[:-1]
        self.utilities = self.utility_matrix.flatten().reshape(-1, self.num_players)
        self.node_adjacency_matrix = None
        self.edge_weight_matrix = None
        self.create_edge_matrix()
        self.graph = None
        self.create_graph()
        self.potentials = None
        self.potential_matrix = None

    def draw_graph(self, circular_layout=False):
        node_positions = {}
        if self.num_players > 3 or circular_layout:
            node_positions = nx.circular_layout(self.graph)
        else:
            for idx in range(len(self.utilities)):
                actions = actions_from_idx(idx, self.num_actions)
                n_dim_positions = actions_from_idx(idx, self.num_actions)
                n_dim_positions[-2] = self.num_actions[-2] - 1 - n_dim_positions[-2]
                if len(n_dim_positions) == 3:
                    n_dim_positions[-1] += n_dim_positions[0] * 0.2
                    n_dim_positions[-2] += n_dim_positions[0] * 0.2
                node_positions[tuple(actions)] = [n_dim_positions[-1], n_dim_positions[-2]]
        nx.draw_networkx_nodes(self.graph, pos=node_positions, alpha=0.2)
        node_labels = {}
        for node in node_positions.keys():
            node_labels[node] = "".join(map(str, node))
        nx.draw_networkx_labels(self.graph, pos=node_positions, labels=node_labels, font_size=10, font_weight="bold")
        edges = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d["weight"] == 0]
        nx.draw_networkx_edges(self.graph, pos=node_positions, edgelist=edges, arrowstyle="->", arrowsize=20, width=3, alpha=0.5, edge_color="#F4D03F")
        edges = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d["weight"] != 0]
        nx.draw_networkx_edges(self.graph, pos=node_positions, edgelist=edges, arrowstyle="->", arrowsize=20, width=1, alpha=0.5, edge_color="r")
        plt.show()

    def create_graph(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([tuple(actions_from_idx(idx, self.num_actions)) for idx in range(len(self.utilities))])
        for from_node_idx in range(len(self.utilities)):
            weighted_edges = []
            from_node_actions = actions_from_idx(from_node_idx, self.num_actions)
            for to_node_idx in range(len(self.utilities)):
                to_node_actions = actions_from_idx(to_node_idx, self.num_actions)
                if self.node_adjacency_matrix[from_node_idx][to_node_idx]:
                    weighted_edges.append((tuple(from_node_actions), tuple(to_node_actions), self.edge_weight_matrix[from_node_idx][to_node_idx]))
            self.graph.add_weighted_edges_from(weighted_edges)

    def create_edge_matrix(self):
        self.node_adjacency_matrix = []
        self.edge_weight_matrix = []
        for from_node_idx, from_node in enumerate(self.utilities):
            edge_adjacency = [0 for _ in range(len(self.utilities))]
            edge_weights = [0 for _ in range(len(self.utilities))]
            from_actions = actions_from_idx(from_node_idx, self.num_actions)
            for player_idx, n_actions in enumerate(self.num_actions):
                for a in range(n_actions):
                    if from_actions[player_idx] == a:
                        continue
                    to_actions = [*from_actions]
                    to_actions[player_idx] = a
                    to_node_idx = idx_from_actions(to_actions, self.num_actions)
                    to_node = self.utilities[to_node_idx]
                    if from_node[player_idx] - to_node[player_idx] > 0:
                        continue
                    edge_weights[to_node_idx] = to_node[player_idx] - from_node[player_idx]
                    edge_adjacency[to_node_idx] = 1
            self.edge_weight_matrix.append(edge_weights)
            self.node_adjacency_matrix.append(edge_adjacency)
        self.edge_weight_matrix = np.array(self.edge_weight_matrix)
        self.node_adjacency_matrix = np.array(self.node_adjacency_matrix)

    def remove_weak_improvement_cycles(self):
        scc = nx.strongly_connected_components(self.graph)
        cycle = {}
        while cycle is not None:
            try:
                cycle = next(scc)
            except StopIteration:
                cycle = None
            if cycle is None or len(cycle) < 3:
                continue
            cycle = list(cycle)
            for from_node in cycle:
                for to_node in cycle:
                    from_node = tuple(from_node)
                    to_node = tuple(to_node)
                    if from_node == to_node or not self.graph.has_edge(from_node, to_node):
                        continue
                    self.graph[from_node][to_node]["weight"] = 0

    def contract_zero_edge_nodes(self, graph):
        mappings = {}
        for idx, entry in enumerate(self.node_adjacency_matrix):
            from_node = tuple(actions_from_idx(idx, self.num_actions))
            while from_node in mappings.keys():
                from_node = mappings[from_node]
            to_nodes = [tuple(actions_from_idx(i, self.num_actions)) for i in np.nonzero(entry)[0]]
            for to_node in to_nodes:
                while to_node in mappings.keys():
                    to_node = mappings[to_node]
                if from_node == to_node:
                    continue
                if graph[from_node][to_node]["weight"] == 0:
                    if "contains" not in graph.nodes[from_node].keys():
                        graph.nodes[from_node]["contains"] = []
                    if "contains" in graph.nodes[to_node].keys():
                        graph.nodes[from_node]["contains"]\
                            .extend(graph.nodes[to_node]["contains"])
                    mappings[to_node] = from_node
                    graph.nodes[from_node]["contains"].append(to_node)
                    for node in graph.nodes:
                        if graph.has_edge(node, from_node) and graph.has_edge(node, to_node):
                            graph[node][from_node]["weight"] = max(graph[node][from_node]["weight"], graph[node][to_node]["weight"])
                    nx.contracted_nodes(graph, from_node, to_node, self_loops=False, copy=False)

    def create_potential_matrix(self, debug=False):
        graph = self.graph.copy()
        self.contract_zero_edge_nodes(graph)
        if debug:
            nx.draw_networkx(graph)
            plt.show()
        ts_nodes = [node for node in nx.topological_sort(graph)]
        potentials = [0 for _ in ts_nodes]
        self.potentials = [0 for _ in range(0, len(self.utilities))]
        for idx, node in enumerate(ts_nodes):
            candidates = []
            for i in range(0, idx):
                if graph.has_edge(ts_nodes[i], node):
                    candidates.append(potentials[i] + graph[ts_nodes[i]][node]["weight"])
            if len(candidates) > 0:
                potentials[idx] = max(candidates)
                self.potentials[idx_from_actions(node, self.num_actions)] = potentials[idx]
                if "contains" in graph.nodes[node].keys():
                    for sub_node in graph.nodes[node]["contains"]:
                        self.potentials[idx_from_actions(sub_node, self.num_actions)] = potentials[idx]
        self.potential_matrix = np.array(self.potentials).reshape(self.utility_matrix.shape[:-1])
