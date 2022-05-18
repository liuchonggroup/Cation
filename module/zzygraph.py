"""
python v3.7.9
@Project: code
@File   : zzygraph.py
@Author : Zhiyuan Zhang
@Date   : 2021/12/7
@Time   : 20:30
"""
import copy
from typing import Union, Dict, Hashable, Tuple, List


class Graph(object):
    """    Graph in topology    """

    def __init__(self, *nodes: Union[Hashable, Tuple[Hashable, Dict]],
                 edges_info: Union[Tuple[Hashable, Dict, Tuple[Hashable, Dict]],
                                   List[Tuple[Hashable, Dict, Tuple[Hashable, Dict]]]] = None):
        """

        :param nodes:
        :param edges_info:
        """

        self.name = "Unnamed"
        self._nodes = []
        self._edges = []

        # Initializing the nodes
        for node_info in nodes:
            self.add_node_by_info(node_info)

        # If given edges isn't a list, Nest a list in the outer layer
        if not edges_info:
            edges_info = []
        elif not isinstance(edges_info, Tuple):
            edges_info = [edges_info]

        for edge_info in edges_info:
            self.add_edge_by_info(edge_info)

    @property
    def nodes(self):
        """    Return all of nodes by a dict, in which the keys are the nodes' names and values are nodes    """
        return NodeDict({node.name: node for node in self._nodes})

    @property
    def edges(self):
        """
        Return all of edge by a dict{dict} , in which the first key are the out nodes' names,
        second keys are the in nodes' names, and values are edges
        """
        return EdgeDict(self._edges)

    @property
    def adj(self):
        """"""
        return ((e.out_node.name, e.in_node.name)for e in self._edges)

    def add_edge_by_info(self, edge_info, replace=False, report_replace=True):
        """    Initializing used a single edge information    """
        if isinstance(edge_info, Tuple) and len(edge_info) in [2, 3]:
            oni = edge_info[0]  # out node information
            ini = edge_info[1]  # into node information

            out_node = self.add_node_by_info(oni)
            in_node = self.add_node_by_info(ini)

            # initializing Edge
            attr_dict = edge_info[2] if len(edge_info) == 3 else {}
            if not isinstance(attr_dict, Dict):
                raise TypeError("the given attribute's information is expected to be a Dict!")
            edge = Edge(out_node, in_node, **attr_dict)
            edge.graph = self
        else:
            raise ValueError("the given edge info is expected to be Tuple as the format,"
                             "(out_node_info, in_node_info, edge_attr[optional])")

        if edge not in self._edges:
            self._edges.append(edge)
        else:
            new_attr = edge.attr
            edge = self.edges[edge.out_node.name, edge.in_node.name][0]
            old_attr = edge.attr
            if replace:
                edge.set_attr(**new_attr)
                if report_replace:
                    print(f"Edge {edge.out_node.name}-{edge.in_node.name} attr changed:\n"
                          f"old attr: {old_attr}\n"
                          f"solubility attr: {edge.attr}")

        return edge

    def add_node_by_info(self, node_info, replace=False, replace_report=True):
        """
        Create a Node, whose dependent graph is this graph
        :param node_info: Node's information
        :param replace: If True and add_into_graph is True too,
                        replace old node by the solubility created when a same node in the graph's nodes list.
                        Or, a error will raise.
        :param replace_report: If True, show the difference of node.attr between before and after replace.
        :return:
        """
        if isinstance(node_info, tuple):
            if len(node_info) == 2 and isinstance(node_info[0], Hashable) and isinstance(node_info[1], Dict):
                node = Node(node_info[0], **node_info[1], graph=self)
            else:
                raise TypeError("the Tuple save node info is expected to the format with (Hashable, Dict)"
                                f"However, the got is {(type(o) for o in node_info)}")
        elif isinstance(node_info, Hashable):
            node = Node(node_info, graph=self)
        else:
            raise TypeError("the given node name is not Hashable")

        if node not in self._nodes:
            self._nodes.append(node)
        else:
            new_attr = node.attr
            node = self.nodes[node.name]
            old_attr = node.attr
            if replace:
                node.set_attr(**new_attr)
                if replace_report:
                    print(f"Node {node.name} changed:\n"
                          f"old attr: {old_attr}\n"
                          f"solubility attr: {node.attr}")

        return node

    def add_node_by_node(self, node):
        """"""
        if not isinstance(node, Node):
            raise TypeError("the argument must be a Node class")
        new_node = copy.deepcopy(node)
        new_node.graph = self

        if new_node not in self._nodes:
            self._nodes.append(new_node)
        else:
            raise ValueError(f"the Node named {new_node.name} have been exist in the graph!")

    @property
    def isolate_nodes(self):
        """    Return a list of isolate nodes in the graph    """
        return [n for n in self._nodes if n.is_isolate]

    def original_nodes(self):
        """    Return a list of original nodes in the graph    """
        return [n for n in self._nodes if n.is_origin]

    def terminal_nodes(self):
        """    Return a list of terminal nodes in the graph    """
        return [n for n in self._nodes if n.is_terminate]

    def remove_nodes(self, *names: str):
        """    Remove nodes by their names    """
        nodes = self.nodes
        edges = self.edges
        for name in names:
            self._nodes.remove(nodes[name])
            for edge in edges[name]:
                self._edges.remove(edge)

    def remove_edges(self, *edges: Tuple):
        """"""
        if all(isinstance(e, Tuple) and len(e) == 2 for e in edges):
            edges = self.edges
            for node_pairs in edges:
                edge = edges[node_pairs[0], node_pairs[1]]
                self._edges.remove(edge)
        else:
            raise KeyError("remove edges need to give a pairs as [in_node.name, out_node.name] exactly!")

    def set_name(self, name):
        """"""
        self.name = name

    def __repr__(self):
        """"""
        return f"Graph({self.name}, Nodes:{len(self._nodes)}, Edges:{len(self._edges)})"


class Node(object):
    """    a Node or Vertex in topology    """
    def __init__(self, name: Hashable, *, graph: Graph = None, coordinate: Tuple = None, **attr):
        """
        Initialize a Node.
        It is not recommended to initialize directly, but to create a Node Using Graph.create_node_by_info
        :param name:
        :param graph:
        :param coordinate:
        :param attr:
        """
        # Check parameters
        if not isinstance(name, Hashable):
            raise TypeError(f"the name of Node is not hashable")
        if not isinstance(attr, Dict):
            raise TypeError(f"The type of Node.attr is expected to be a dict, but get a {type(attr)} instead")

        self.name = name
        self._attr = attr
        self.coordinate = coordinate
        self.graph = graph

    def __eq__(self, other):
        """    Determine whether this node is equal with other    """
        if not isinstance(other, Node):
            raise TypeError(f"{type(other)} class is not comparable with this Node class!")

        if self.name == other.name and self.graph is other.graph:
            return True
        return False

    def __hash__(self):
        """"""
        return hash(f"{self.graph}{self.name}")

    def __repr__(self):
        """"""
        return f"Node({self.name}, Attr: {self._attr})"

    @property
    def _basic_info(self):
        """   Output basic information of this nodes to a dict    """
        return {"name": self.name, "attr": self._attr, "coordinate": self.coordinate, "graph": self.graph}

    @property
    def predecessors(self):
        """"""
        return [self.graph.nodes[n] for n in self.predecessors_names]

    def add_attr(self, *, coordinate=None, **attr):
        """    Add a solubility attributes into Node, if the attribute is exist, replace it to the solubility    """
        # check arguments
        forbidden_names = ["graph", "name", "successors", "predecessors"]
        for name in forbidden_names:
            if name in attr:
                raise KeyError(f"the {name} attr can't be set!")

        if coordinate and isinstance(coordinate, Tuple):
            self.coordinate = coordinate
        for name, value in attr.items():
            self._attr[name] = value

    def add_successor(self, name: Hashable, successor_attr: Dict = None,
                      edge_attr: Dict = None, edge_weight: float = None):
        """"""
        if self.graph:
            graph = self.graph
        else:
            raise ValueError("this Node do not in a graph")

        edge = graph.add_edge_by_info((self.name, name))
        if edge_attr:
            edge.set_attr(**edge_attr)
        if edge_weight:
            edge.set_weight(edge_weight)

        successor = edge.in_node
        if successor_attr:
            successor.set_attr(**successor_attr)

        return successor

    def add_predecessor(self, name: Hashable, successor_attr: Dict = None,
                        edge_attr: Dict = None, edge_weight: float = None):
        """"""
        if self.graph:
            graph = self.graph
        else:
            raise ValueError("this Node do not in a graph")

        edge = graph.add_edge_by_info((name, self.name))
        if edge_attr:
            edge.set_attr(**edge_attr)
        if edge_weight:
            edge.set_weight(edge_weight)

        predecessor = edge.in_node
        if successor_attr:
            predecessor.set_attr(**successor_attr)

        return predecessor

    @property
    def attr(self):
        """    Get nodes attributes dict    """
        return self._attr

    @property
    def edges(self):
        """"""
        return list(self.successors_edge) + list(self.predecessor_edge)

    @property
    def is_isolate(self):
        """    Judge whether the node is isolate    """
        if not self.neighbours_name:
            return True
        return False

    @property
    def is_middle(self):
        """    Judge weather the node is in middle    """
        return not self.is_origin and not self.is_terminate and not self.is_isolate

    @property
    def is_origin(self):
        """    Judge whether the node is original    """
        if not self.predecessors_names and self.successors_name:
            return True
        return False

    @property
    def is_terminate(self):
        """    Judge whether the node is terminal    """
        if not self.successors_name and self.predecessors_names:
            return True
        return False

    @property
    def neighbours(self):
        """"""
        return [self.graph.nodes[n] for n in self.neighbours_name]

    @property
    def neighbours_name(self):
        """"""
        return list(set(self.successors_name).union(set(self.predecessors_names)))

    @property
    def predecessor_edge(self):
        """"""
        return [self.graph.edges[pn, self.name] for pn in self.predecessors_names]

    @property
    def predecessors_names(self):
        """"""
        return [p[0] for p in self.graph.adj if p[1] == self.name]

    def set_attr(self, *, coordinate=None, **attr):
        """    Set node's attributes    """
        # check arguments
        forbidden_names = ["graph", "name", "successors", "predecessors"]
        for name in forbidden_names:
            if name in attr:
                raise KeyError(f"the {name} attr can't be set using this method!")
        if coordinate and isinstance(coordinate, Tuple):
            self.coordinate = coordinate
        self._attr = attr

    @property
    def successors(self):
        """"""
        return [self.graph.nodes[n] for n in self.successors_name]

    @property
    def successors_edge(self):
        """"""
        return [self.graph.edges[self.name, sn] for sn in self.successors_name]

    @property
    def successors_name(self):
        """"""
        return [p[1] for p in self.graph.adj if p[0] == self.name]


class Edge(object):
    """
    Directed Edge in topology
    The Edge must be in a Graph instance, that a isolated Edge is meaningless
    """
    def __init__(self, out_node: Node, in_node: Node, *, weight: Union[int, float, complex] = 1.0,
                 graph: Graph = None, **attr):
        """"""
        self.out_node = out_node
        self.in_node = in_node
        self.weight = weight
        self._attr = attr
        self.graph = graph

    def __eq__(self, other):
        """    Two Edges are same with each other, if their out_node and in_node are exactly same    """
        if not isinstance(other, Edge):
            raise TypeError(f"The {type(other)} is not comparable with Edge class")
        if self.out_node == other.out_node and self.out_node == other.in_node:
            return True
        return False

    def __hash__(self):
        """"""
        return hash(f"{self.out_node}{self.in_node}")

    @property
    def attr(self):
        """    Return the Dict of Edge's attributes    """
        return self._attr

    def set_attr(self, *, weight=None, **attr):
        """    Set Edge's attributes    """
        # check arguments
        forbidden_names = ["graph", "out_node", "in_node"]
        for name in forbidden_names:
            if name in attr:
                raise KeyError(f"the {name} attr can't be set using this method!")

        self._attr = attr
        if weight:
            self.set_weight(weight)

    def set_weight(self, weight: float):
        """"""
        if not isinstance(weight, float):
            raise TypeError("the Edge's weight need to be float!")
        self.weight = weight

    @property
    def nodes(self):
        """"""
        return self.out_node, self.in_node

    def __repr__(self):
        """"""
        return f"Edge('{self.out_node.name}','{self.in_node.name}')"


class EdgeDict(object):
    """    A class is used to get Edges conveniently in  a graph    """
    def __init__(self, edges: List[Edge]):
        if not isinstance(edges, List) or any(not isinstance(e, Edge) for e in edges):
            raise TypeError("This class is only used to get Edges in a graph!")
        self.edges = edges

    def __getitem__(self, nodes_names: Union[Hashable, Tuple[Hashable]]):
        """
        Get items by subscript
        :param nodes_names: This parameter can receive up to two values,
                           the first of which is the out_node's name;
                           the second of which is the in_node's name.
        :return:
        """
        # Check parameters
        if not isinstance(nodes_names, (Tuple, List, set)):
            nodes_names = [nodes_names]

        if len(nodes_names) > 2:
            raise ValueError(f"This parameter can receive up to two values, the first of which is the out_node's name;"
                             f"the second of which is the in_node's name.\n"
                             f"However, the parameter got {len(nodes_names)} values instead!")

        if not nodes_names:
            return self.edges

        if len(nodes_names) == 1:
            return [e for e in self.edges if e.out_node.name in nodes_names or e.in_node.name in nodes_names]

        if len(nodes_names) == 2:
            list_edges = []
            for idx, node_names in enumerate(nodes_names):
                if not node_names or node_names == slice(None, None, None):
                    edge = self.edges
                elif isinstance(node_names, Tuple):
                    edge = [e for e in self.edges if node_names and e.out_node.name in node_names] if idx == 0 \
                        else [e for e in self.edges if node_names and e.in_node.name in node_names]
                elif isinstance(node_names, Hashable):
                    edge = [e for e in self.edges if node_names and e.out_node.name == node_names] if idx == 0 \
                        else [e for e in self.edges if node_names and e.in_node.name == node_names]
                else:
                    raise TypeError("The subscript must be Hashable or a Sequence of Hashable object!")

                list_edges.append(edge)

            return list(set(list_edges[0]).intersection(set(list_edges[1])))

    def __len__(self):
        """"""
        return len(self.edges)

    def __repr__(self):
        """"""
        return f"{self.edges}"

    def items(self):
        """"""
        return ((e.out_node.name, e.in_node.name) for e in self.edges)

    def keys(self):
        """"""
        return ((e.out_node.name, e.in_node.name) for e in self.edges)

    def values(self):
        """"""
        return (e for e in self.edges)


class NodeDict(Dict):
    """"""

    def __repr__(self):
        """"""
        return f"{list(self.values())}"
