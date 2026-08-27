"""
GraphRAG layer (optional, stretch goal — implement after the core pipeline works).

Vector/hybrid search treats each chunk independently, so it struggles with
questions that require traversing relationships: "which of Apple's named
suppliers are also mentioned in its litigation disclosures?" needs to connect
entities across sections and documents. This module extracts (entity,
relation, entity) triples from chunks with an LLM, builds a NetworkX graph,
and answers multi-hop questions by traversing it rather than re-ranking chunks.

This is deliberately the most involved/optional piece — build it last, and
in an interview you can speak to the tradeoff: graph extraction is expensive
and noisy, so it's worth it only for the subset of queries that are genuinely
relational (query_router.py flags these as "multi_hop").
"""
import json
import pickle
from dataclasses import dataclass

import networkx as nx

from src.config import GRAPH_PATH
from src.llm import get_llm
from src.ingest.chunking import Chunk

EXTRACTION_PROMPT = """Extract entity relationships from this SEC filing excerpt.
Focus on: companies, subsidiaries, suppliers/vendors, executives, risk factors,
legal proceedings, and products. Return ONLY a JSON list of triples:

[{{"source": "...", "relation": "...", "target": "..."}}]

Text:
{text}"""


@dataclass
class Triple:
    source: str
    relation: str
    target: str
    chunk_id: str


class EntityGraphBuilder:
    def __init__(self):
        self.llm = get_llm()
        self.graph = nx.MultiDiGraph()

    def extract_triples(self, chunk: Chunk) -> list[Triple]:
        response = self.llm.invoke(EXTRACTION_PROMPT.format(text=chunk.text[:1500]))
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [
            Triple(source=i["source"], relation=i["relation"], target=i["target"], chunk_id=chunk.chunk_id)
            for i in items if all(k in i for k in ("source", "relation", "target"))
        ]

    def build(self, chunks: list[Chunk], sample_rate: float = 1.0) -> None:
        """sample_rate < 1.0 lets you extract from a subset of chunks to control
        LLM cost during development — extraction is the most expensive step here."""
        import random
        sampled = chunks if sample_rate >= 1.0 else random.sample(chunks, int(len(chunks) * sample_rate))

        for chunk in sampled:
            triples = self.extract_triples(chunk)
            for t in triples:
                self.graph.add_edge(t.source, t.target, relation=t.relation, chunk_id=t.chunk_id)

        print(f"[entity_graph] built graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges from {len(sampled)} chunks")

    def save(self) -> None:
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self) -> None:
        with open(GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)

    def find_paths(self, entity_a: str, entity_b: str, max_hops: int = 3) -> list[list[str]]:
        """Find relationship paths between two entities — the core of graph-based
        multi-hop answering. Returns node paths; look up edge 'relation' attrs to narrate them."""
        try:
            paths = list(nx.all_simple_paths(self.graph, entity_a, entity_b, cutoff=max_hops))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def neighbors_with_relations(self, entity: str) -> list[tuple[str, str]]:
        """Return (relation, neighbor) pairs for one-hop exploration around an entity."""
        if entity not in self.graph:
            return []
        results = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            results.append((data.get("relation", "related_to"), target))
        return results
